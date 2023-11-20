#%%
import numpy as np
from pr3_utils import *
from scipy.linalg import expm,block_diag
from tqdm import tqdm 
from os import mkdir
from os.path import exists
import time

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))

def flip_roll(T):
	if T.ndim==3:
		T_flip = np.tile(np.eye(4) ,(T.shape[0],1,1))
		T_flip[:,1,1]=-1
		T_flip[:,2,2]=-1
		return T @ T_flip 
	else: 
		T_flip = np.eye(4)
		T_flip[1,1]=-1
		T_flip[2,2]=-1 
		return T @ T_flip
def triangulate(z1, z2,b):
	'''
	z1, z2: 2x1, Camera pixel coordinate 
	b: scalar, stereo baseline
	'''
	e3 = np.array([0,0,1])[:,np.newaxis]
	p = np.array([b,0,0])[:, np.newaxis]
	z1h,z2h = np.ones((3,1)), np.ones((3,1))
	z1h[:2,0] = z1 
	z2h[:2,0] = z2 
	a1 = p - (e3.T @ p) * z2h 
	b1 = z1h- (e3.T @z1h) * z2h 
	result = np.ones((4,1))
	result[:3] = (a1.T @ a1) / (a1.T @ b1) * z1h 
	return result 

def px2cam(stereo,K,b): 
	'''
	@Input
		K, b: stereo camera model parameters (intrinstic matrix, baseline)
		stereo: 4 x n, visual feature point coordinates in stereo images at each time stamp
	@Output 
		homogeneous coordinate in camera frame: 4 x n
	'''
	z = K[0,0]*b / (stereo[0,:]-stereo[2,:])
	x = (stereo[0,:]-K[0,2])*z / K[0,0]
	y = (stereo[1,:]-K[1,2])*z / K[1,1]
	return np.vstack((x,y,z,np.ones(stereo.shape[1])))


def update_cov(cov_ldmk_t,new_data,correlation=None):
	'''
	@Input: 
		cov_ldmk_t: M x M, The previous landmark covariances 
		new_data: N x N, newly initialized landmark covariances
		correlation: N x M, the block at the lower left corner
	@Output: 
		new_cov: (M+N) x (M+N)
	'''
	prevM, addedM = cov_ldmk_t.shape[0], new_data.shape[0]
	if correlation is not None: 
		return np.block([[cov_ldmk_t,correlation.T],[correlation, new_data]])
	else:
		return np.block([[cov_ldmk_t,np.zeros((prevM,addedM))],[np.zeros((addedM,prevM)), new_data]])

def init_Ks(K,b):
	Ks = np.zeros((4,4))
	Ks[:2,:3] = K[:2,:]
	Ks[2:,:3] = K[:2,:]
	Ks[2,3] = -K[0,0]*b
	return Ks

def ldmk_H_and_z(Ks,world_T_rob,rob_T_cam,landmarks,PhT):
	'''
	@Input:
		Ks: 4 x 4, stereo intrisic matrix
		world_T_rob: 4 x 4
		rob_T_cam: 4 x 4
		landmarks: Nt x 4, 
		PhT: 4 x 3
	@Output: 
		jacob: Jacobian for all features: Nt x 4 x 3
		pred_z: Predicted observation for all features: Nt x 4
	'''
	Nt = landmarks.shape[0]
	cam_T_world = inversePose(np.tile(world_T_rob@rob_T_cam,(Nt,1,1)))
	Ks = np.tile(Ks,(Nt,1,1))
	PhT = np.tile(PhT,(Nt,1,1))
	pred_opt = (cam_T_world @ landmarks[...,np.newaxis]).squeeze(axis=-1)
	jacob = Ks @ projectionJacobian(pred_opt) @ cam_T_world @ PhT
	pred_z = (Ks @ projection(pred_opt)[...,np.newaxis]).squeeze(axis=-1)
	return jacob,pred_z

def Ht(Hblocks_t,obs_feat_indices,M):
	'''
	@Input: 
		Hblocks_t: Nt x 4 x 3, H blocks for all features at time t
		obs_feat_indices: Nt, The indices of the observed features
		M: scalar, current stored landmarks
	@Output: 
		Htout: (4*Nt) x (3*M), final Jacobian needed
	'''
	block_diaged = block_diag(*Hblocks_t)
	Htout = np.zeros((4*Hblocks_t.shape[0],3*M))
	indices = np.hstack([(np.arange(i*3,i*3+3)) for i in obs_feat_indices])
	Htout[:,indices] = block_diaged
	return Htout 

def rob_ldmk_H(Ks,mu_rob,rob_T_cam,landmarks):
	'''
	@Input:
		Ks: 4 x 4, stereo intrisic matrix
		mu_rob: 4 x 4
		rob_T_cam: 4 x 4
		landmarks: Nt x 4, 
	@Output: 
		jacob: Jacobian for all features: Nt x 4 x 6
	'''
	Nt = landmarks.shape[0]
	rob_T_world = inversePose(np.tile(mu_rob,(Nt,1,1)))
	cam_T_rob = inversePose(np.tile(rob_T_cam,(Nt,1,1)))
	cam_T_world = cam_T_rob @ rob_T_world
	Ks = np.tile(Ks,(Nt,1,1))
	pred_opt = (cam_T_world @ landmarks[...,np.newaxis]).squeeze(axis=-1)
	intermediate = homocrdHat((rob_T_world @ landmarks[...,np.newaxis]).squeeze(axis=-1))
	jacob = -Ks @ projectionJacobian(pred_opt) @ cam_T_rob @ intermediate
	return jacob

def compare_map(imu_tra_dr,ldmk_dr,imu_tra,ldmk,
				save_path=None, fname='', show=True, xlim=None, ylim=None):
	fig,ax = plt.subplots(figsize=(7,5))
	ax.plot(imu_tra_dr[:,0,3],imu_tra_dr[:,1,3],color='orange',label='Trajectory DR')
	ax.scatter(ldmk_dr[0,:].T,ldmk_dr[1,:].T,color='r',label='Landmarks DR',s=0.5)
	
	ax.plot(imu_tra[:,0,3],imu_tra[:,1,3],color='b',label='Trajectory SLAM')
	ax.scatter(ldmk[0,:].T,ldmk[1,:].T, color='g', label='Landmarks SLAM',s=0.5)

	ax.scatter(imu_tra_dr[0,0,3],imu_tra_dr[0,1,3],marker='s',label="start")
	ax.scatter(imu_tra_dr[-1,0,3],imu_tra_dr[-1,1,3],marker='o',label="end DR")
	ax.scatter(imu_tra[-1,0,3],imu_tra[-1,1,3],marker='o',label="end SLAM")
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.axis('equal')	
	if xlim is None:
		xlim = [-1300,800]
		ylim = [-1000,800]
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.grid(False)
	ax.legend(loc='upper right')
	ax.set_title("Comparison of dead reckoning and SLAM")
	plt.tight_layout()
	if save_path is not None: 
		fname = save_path+"Comparison%s/"%fname
		if not exists(fname):
			mkdir(fname)
		plt.savefig(fname+"combined.png")
		plt.pause(0.01)
	if show:
		plt.show(block=False)
	return fig, ax
#%%
# ----------------------(a) IMU Localization via EKF Prediction---------------------
def localization_dr(imu_pose,t,save_path=None, show=True,block=False):
	init_pose = np.eye(4)
	imu_trajectory_dr = np.zeros(imu_pose.shape) 
	imu_trajectory_dr[0,:,:] = init_pose 
	for i in range(0,t.shape[1]-1): 
		imu_trajectory_dr[i+1,:,:] = imu_trajectory_dr[i,:,:] @ imu_pose[i,:,:]
	fig, ax = visualize_trajectory_2d(imu_trajectory_dr.transpose((1,2,0)), show_ori=True, show=show, block=block)
	if save_path is not None: 
		fname = save_path+"LocalizationOnly/"
		if not exists(fname):
			mkdir(fname)
		ax.set_title("Dead reckoning localization of the robot")
		plt.savefig(fname+"dr.png")
		plt.pause(0.01)
	return imu_trajectory_dr
#%%--------------------(b) Landmark Mapping via EKF Update----------------------- 
def mapping_dr(imu_trajectory_dr,t,features, rob_T_cam, K, b, Ks,
	       		save_path=None,show=True,fname='', save_freq=2,
	       		xlim=None, ylim=None):
	if save_path is not None: 
		fname = save_path+"MapOnly%s/"%fname
		if not exists(fname):
			mkdir(fname)

	N_feat_total = features.shape[1]
	t_len = t.shape[1]
	mu_ldmk_t = np.ones((4,N_feat_total))
	correspondence = np.zeros(N_feat_total,dtype=bool)
	dist_thresh = 1000 
	ldmk_obs_noise = 2
	Vnoise = 5
	cov_ldmk_t  = np.eye(0) 
	PhT = np.hstack((np.eye(3),np.zeros((3,1)))).T

	tstart = tic()
	graph_obs = visualize_trajectory_with_landmarks_2d(imu_trajectory_dr.transpose((1,2,0)),
				landmarks=mu_ldmk_t,show_ori=False,block=False,show=show,xlim=xlim,ylim=ylim)
	for i in tqdm(range(t_len)): 
		# Determine the indices of features ovserved in the timestamp
		Nt = (features[0,:,i]!=-1)

		# Pixel coordinates of the observed features
		pixel_coords = features[:,Nt,i]
		N_indices = pixel_coords.shape[1] # Number of observed indices
		# Compute the world coordinates of the observed features with shape Nt x 4
		world_coords = (np.tile(imu_trajectory_dr[i,:,:]@rob_T_cam,(N_indices,1,1)) @ 
						(px2cam(pixel_coords,K,b).T)[:,:,np.newaxis]).squeeze(axis=-1)
		
		# Some observations can be very far away and thus outliers to be rejected
		Nt_numerical_ind = np.where(Nt)[0]
		robot_pos = imu_trajectory_dr[i,:3,3]
		rejected = np.linalg.norm(world_coords[:,:3]-robot_pos[np.newaxis,:], axis=1)>=dist_thresh
		Nt[Nt_numerical_ind[rejected]] = False 
		Nt_numerical_ind = Nt_numerical_ind[~rejected]
		world_coords = world_coords[~rejected,:]
		pixel_coords = pixel_coords[:,~rejected]
		N_indices = pixel_coords.shape[1] # Number of observed indices
		if N_indices==0:
			continue

		# Compute the indices that are first seen and expand covariance and
		# mean of the new landmark observations accordingly
		indices_first_seen = np.logical_and(Nt, correspondence==0)
		correspondence[indices_first_seen]=1
		Nt_numerical_ind_1stseen = np.where(indices_first_seen)[0]
		N_indices_new = np.count_nonzero(indices_first_seen)
		new_mean_ind = np.where(Nt_numerical_ind[:,None] == Nt_numerical_ind_1stseen[None,:])[0]
		mu_ldmk_t[:,indices_first_seen] = world_coords[new_mean_ind,:].T
		new_ldmk_cov = np.eye(3*N_indices_new)*ldmk_obs_noise
		cov_ldmk_t = update_cov(cov_ldmk_t,new_ldmk_cov) 


		# Select the updating indices 
		Nt_numerical_ind = np.hstack([(np.arange(i*3,i*3+3)) for i in Nt_numerical_ind])
		cropped_cov = cov_ldmk_t[np.ix_(Nt_numerical_ind,Nt_numerical_ind)]
		# Update step with Kalman gain K_{t+1}, mean mu_{t+1}, and covariance Sigma_{t+1}
		Hblocks,pred_obs = ldmk_H_and_z(Ks,imu_trajectory_dr[i,:,:],rob_T_cam,world_coords,PhT)
		H_t = block_diag(*Hblocks)
		intermediate = cropped_cov @ H_t.T
		KalmanG = intermediate @ np.linalg.inv(H_t @ intermediate + np.eye(4*N_indices)*Vnoise)
		mu_ldmk_t[:3,Nt] += (KalmanG @ (pixel_coords - pred_obs.T).reshape((-1,1),order='F')).reshape((3,-1),order='F')
		cov_ldmk_t[np.ix_(Nt_numerical_ind,Nt_numerical_ind)] = (np.eye(N_indices*3)-KalmanG@H_t) @ cropped_cov 

		if i%save_freq==0: 	
			update_graph(imu_trajectory_dr.transpose((1,2,0)),mu_ldmk_t,i,graph_obs,show=show)
			if save_path is not None:
				plt.savefig(fname+"trajectory_%d.png"%i)

	toc(tstart,'mapping only iterations')
	return mu_ldmk_t

#%%  ----------------------------------(c) Visual-Inertial SLAM-------------------------------
def slam(imu_pose, imu_adpose, t,features, rob_T_cam, K, b, Ks,
	       		save_path=None,show=True,fname='', save_freq=2,
				Wnoise=(1e-1, 1e-5), Vnoise=5, ldmknoise=2, 
	       		xlim=None, ylim=None):
	if save_path is not None: 
		fname = save_path+"SLAM%s/"%fname
		if not exists(fname):
			mkdir(fname)
	N_feat_total = features.shape[1]
	N_feat_seen = 0
	t_len = t.shape[1]
	correspondence = np.zeros(N_feat_total,dtype=bool)
	dist_thresh = 1000 
	PhT = np.hstack((np.eye(3),np.zeros((3,1)))).T

	# Noise tuning and initialization
	# Wnoise_lin, Wnoise_ang = 1e-1, 1e-5
	Wnoise_lin, Wnoise_ang = Wnoise[0], Wnoise[1]
	Wnoise = np.diag(np.array([Wnoise_lin]*3 + [Wnoise_ang]*3))
	# initRLnoise = np.array([1e-2,1e-2,1e-3]+[Wnoise_ang]*3)
	initRLnoise = np.zeros(6)

	# Means and Covariances initialization
	cov_rob_t = np.diag([1e-2,1e-2,1e-3,1e-5,1e-5,1e-5])
	# cov_rob_t = np.diag([1e-1,1e-1,1e-2,1e-5,1e-5,1e-5])
	mu_rob_t = np.eye(4) 
	cov_ldmk_t  = np.eye(0) 
	mu_ldmk_t = np.ones((4,N_feat_total))
	cov_rbld_t = np.zeros((6,0))

	imu_trajectory = np.zeros(imu_pose.shape) 
	imu_trajectory[0,:,:] = mu_rob_t

	tstart = tic()
	graph_obs = visualize_trajectory_with_landmarks_2d(imu_trajectory.transpose((1,2,0)),
						landmarks=mu_ldmk_t,show_ori=False,block=False,show=show, get_posedata=True,
						xlim=xlim,ylim=ylim)

	for i in tqdm(range(t_len-1)):
		''' Prediciton step'''
		mu_rob_t = mu_rob_t @ imu_pose[i,:,:]
		F_t = imu_adpose[i,:,:]
		cov_rob_t = F_t @ cov_rob_t @ F_t.T + Wnoise
		cov_rbld_t  = F_t @ cov_rbld_t 

		''' Update step'''
		# Determine the indices of features ovserved in the timestamp
		Nt = (features[0,:,i]!=-1)

		pixel_coords = features[:,Nt,i]	# Pixel coordinates of the observed features
		N_indices = pixel_coords.shape[1] # Number of observed indices
		# Compute the world coordinates of the observed features with shape Nt x 4
		world_coords = (np.tile(mu_rob_t@rob_T_cam,(N_indices,1,1)) @ 
						(px2cam(pixel_coords,K,b).T)[:,:,np.newaxis]).squeeze(axis=-1)
		
		# Some observations can be very far away and thus outliers to be rejected
		Nt_numerical_ind = np.where(Nt)[0]
		robot_pos = mu_rob_t[:3,3]
		rejected = np.linalg.norm(world_coords[:,:3]-robot_pos[np.newaxis,:], axis=1)>=dist_thresh
		Nt[Nt_numerical_ind[rejected]] = False 
		Nt_numerical_ind = Nt_numerical_ind[~rejected]
		world_coords = world_coords[~rejected,:]
		pixel_coords = pixel_coords[:,~rejected]
		N_indices = pixel_coords.shape[1]
		if N_indices==0:
			imu_trajectory[i+1,:,:] = imu_trajectory[i,:,:] 
			continue

		# Compute the indices that are first seen and expand covariance and
		# mean of the new landmark observations
		indices_first_seen = np.logical_and(Nt, correspondence==0)
		correspondence[indices_first_seen]=1
		Nt_numerical_ind_1stseen = np.where(indices_first_seen)[0]
		N_indices_new = np.count_nonzero(indices_first_seen)
		N_feat_seen += N_indices_new

		new_mean_ind = np.where(Nt_numerical_ind[:,None] == Nt_numerical_ind_1stseen[None,:])[0]
		mu_ldmk_t[:,indices_first_seen] = world_coords[new_mean_ind,:].T
		new_ldmk_cov = np.eye(3*N_indices_new)*ldmknoise
		cov_ldmk_t = update_cov(cov_ldmk_t,new_ldmk_cov) 
		cov_rbld_t = np.hstack((cov_rbld_t,np.tile(initRLnoise,(3*N_indices_new,1)).T))


		# Select the updating indices 
		cov_slam_t = update_cov(cov_ldmk_t, cov_rob_t, cov_rbld_t)
		Nt_numerical_ind = np.hstack([(np.arange(i*3,i*3+3)) for i in Nt_numerical_ind]+
							np.arange(N_feat_seen*3,N_feat_seen*3+6).tolist())
		cropped_cov = cov_slam_t[np.ix_(Nt_numerical_ind,Nt_numerical_ind)]

		# Update step with Kalman gain K_{t+1}, mean mu_{t+1}, and covariance, Sigma_{t+1}
		Hblocks_ldmk,pred_obs = ldmk_H_and_z(Ks,mu_rob_t,rob_T_cam,world_coords,PhT)
		Hblocks_rbld = rob_ldmk_H(Ks,mu_rob_t,rob_T_cam,world_coords)
		H_ldmk_t = block_diag(*Hblocks_ldmk)
		H_rbld_t = Hblocks_rbld.reshape((-1,Hblocks_rbld.shape[2]))
		H_slam_t = np.hstack((H_ldmk_t,H_rbld_t))
		intermediate = cropped_cov @ H_slam_t.T
		K_slam = intermediate @ np.linalg.inv(H_slam_t @ intermediate + np.eye(4*N_indices)*Vnoise)
		# Separate the landmark kalman gain and robot kalman gain to update individually
		K_ldmk, K_rob = K_slam[:3*N_indices,:], K_slam[3*N_indices:, :]
		res_err_t = (pixel_coords - pred_obs.T).reshape((-1,1),order='F')
		mu_rob_t = mu_rob_t @ axangle2pose((K_rob @ res_err_t).squeeze())
		mu_ldmk_t[:3,Nt] += (K_ldmk @ res_err_t).reshape((3,-1),order='F')
		# Update the covariance matrix of the whole slam covariance, then store the covariances separately
		cov_slam_t[np.ix_(Nt_numerical_ind,Nt_numerical_ind)] = (np.eye((N_indices*3+6))-K_slam@H_slam_t) @ cropped_cov 
		cov_ldmk_t = cov_slam_t[:3*N_feat_seen, :3*N_feat_seen]
		cov_rbld_t = cov_slam_t[3*N_feat_seen:, :3*N_feat_seen]
		cov_rob_t  = cov_slam_t[3*N_feat_seen:, 3*N_feat_seen:]

		imu_trajectory[i+1,:,:] = mu_rob_t 

		if i%save_freq==0: 	
			update_graph(imu_trajectory.transpose((1,2,0)),mu_ldmk_t,i,graph_obs,show=show)
			if save_path is not None:
				plt.savefig(fname+"trajectory_%d.png"%i)
	
	toc(tstart,'testing slam iterations')
	return imu_trajectory, mu_ldmk_t, graph_obs


#%%
if __name__ == '__main__':

	full_evaluate = False
	live_plot_update = True
	save_freq = 2
	subsample_rate = 1

	Wnoise = (1,1e-2)
	Wnoise_list = [(1e-2,1e-5),(1,1e-2)]
	Vnoise_list = [2, 5]
	ldmknoise_list = [0.5,2]
	if full_evaluate:
		combinations = [[i, j, k] for i in Wnoise_list
					for j in Vnoise_list
					for k in ldmknoise_list]
	else: 
		combinations = [[Wnoise_list[0],Vnoise_list[1], ldmknoise_list[1]]]

	for dataset in [3,10]:
		for comb in combinations:
			Wnoise,Vnoise, ldmknoise = comb[0], comb[1], comb[2]
			trial_name='_W%.0E_V_%02d_lm_%.1f'%(Wnoise[0],Vnoise,ldmknoise)
			save_pathname = "../dataset%02dfigs/"%dataset
			if not exists(save_pathname):
				mkdir(save_pathname)
			xlim = [-1200,500] if dataset==3 else [-1300,800]
			ylim = [-500,800] if dataset==3 else [-1000,800]
			data_filename = "../data/%02d.npz"%dataset
			t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(data_filename)
			# full_features = features.copy()
			# features = full_features[:,::subsample_rate,:].copy()
			features = features[:,::subsample_rate,:]
			Ks = init_Ks(K,b)
			rob_T_cam = flip_roll(imu_T_cam)
			imu_twist = np.vstack((linear_velocity,angular_velocity))
			imu_pose = axangle2pose((np.append(np.diff(t),0)[np.newaxis,0] * imu_twist).T)
			imu_adtwist = axangle2adtwist((np.append(np.diff(t),0)[np.newaxis,0] * imu_twist).T)
			imu_adpose = expm(-imu_adtwist)

			imu_tra_dr= localization_dr(imu_pose,t,save_path=save_pathname,show=live_plot_update)
			ldmk_dr = mapping_dr(imu_tra_dr, t, features, rob_T_cam, K, b, Ks, 
								save_path=save_pathname,show=live_plot_update,save_freq=save_freq, fname=trial_name,
								xlim=xlim, ylim=ylim)
			imu_tra, ldmk, graph_obs = slam(imu_pose,imu_adpose,t, features, rob_T_cam, K, b, Ks, 
								save_path=save_pathname,show=live_plot_update,save_freq=save_freq, fname=trial_name,
								Wnoise=Wnoise, Vnoise=Vnoise, ldmknoise=ldmknoise, 
								xlim=xlim, ylim=ylim)
			graph_obs = compare_map(imu_tra_dr,ldmk_dr,imu_tra, ldmk,
								save_path=save_pathname, show=live_plot_update, xlim=xlim, ylim=ylim, fname=trial_name)
	