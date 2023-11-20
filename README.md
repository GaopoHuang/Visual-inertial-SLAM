# Project 3
This project contains Visual-inertial SLAM and its result

## File structure

├── code  
│   ├── training  
│   │   ├── `main.py`: #Visual inertial SLAM implementation 
│   │   ├── `pr3_util.py`:  #Helper functions for graphing and pose kinematics
│   ├── README.md  
├── data  #Contains all the train data
├── dataset03figs
│   │   ├── LocalizationOnly #part(a)
│   │   ├── MapOnly_W1E-02_V_02_lm_05 #part(b)
│   │   ├── SLAM_W1E-02_V_02_lm_05 #part(c) SLAM only
│   │   ├── Comparison_W1E-02_V_02_lm_05 #part(c) SLAM vs trajectory
│   │   └── ...    
├── dataset10figs
│   │   ├── LocalizationOnly #part(a)
│   │   ├── MapOnly_W1E-02_V_02_lm_05 #part(b)
│   │   ├── SLAM_W1E-02_V_02_lm_05 #part(c) SLAM only
│   │   ├── Comparison_W1E-02_V_02_lm_05 #part(c) SLAM vs trajectory
│   │   └── ...    


## Usage 

`main.py` provides the basic visual-inertial SLAM implementation. It can run the dead reckoning prediction and update steps and also visual-inertial SLAM combined. Some parameters that can be set in the main file before running:
> - `full_evaluate`: test different noise tuning.
> - `live_plot_update`: show a live updating plot
> - `subsample_rate`: subsampling the feature files
> - `save_freq`: save figure frequency
> - All noise tuning including W, V, and landmark covariance initialization noise