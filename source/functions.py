#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:22:15 2020

@author: 
"""

class Comparison:
    
  def __init__(self):
    
        super().__init__()
  '''
      MAJ : 26112020
      By : Maurras
      Add new function to only execute the modified IForestASD : Partially update the model
  '''
  #The goal of this function is to execute the models and show the differents results. 
  #It is the function to call when we want to test differents models 
  #with differents values for parameters
  def run_MIForestASD(self, execution_number:int, stream, stream_n_features, window = 100, 
                     estimators = 50, anomaly = 0.5, drift_rate = 0.3, 
                     result_folder="Generated", max_sample=100000, n_wait=200,
                     metrics=['accuracy', 'f1', 'kappa', 'kappa_m', 
                              'running_time','model_size'],
                              n_estimators_updated=0.5, updated_randomly=True):
    
    #from skmultiflow.anomaly_detection import HalfSpaceTrees
    from source.iforestasd_scikitmultiflow import IsolationForestStream
    #from source.iforestasd_adwin_scikitmultiflow import IsolationForestADWINStream
    from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    
    
    # Creation f the result csv
    directory_path = 'results/'+str(result_folder)
    self.check_directory(path=directory_path)
    #nb_update
    
    for i in range(execution_number):
        print("*************************************** Execution N° "+str(i)+"**********************************")
        models = [IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "MAnomalyRate")
                   ]
        
        result_path = directory_path+'/result_Number'+str(i)+'_UR'+str(n_estimators_updated)+'_for_WS'+str(window)+'_NE'+str(estimators)
        result_file_path = result_path+'.csv'
        # Setup the evaluator
        evaluator = EvaluatePrequential(pretrain_size=1, max_samples=max_sample, 
                                        show_plot=True, 
                                        metrics=metrics, batch_size=1, 
                                        output_file = result_file_path,
                                        n_wait = n_wait) 
        # 4. Run the evaluation 
        evaluator.evaluate(stream=stream, model=models, model_names=['M_IFA'
                                                                    ])
        #Save the stats about models updating
        update_file_path = result_path+"updated_count.csv"
#        update_data = []
#        for model in models:
#            update_data.append(model.model_update)
#        self.array_to_csv(data = update_data, file_path=update_file_path)
        self.models_updated_to_csv(models=models, file_path=update_file_path)
        
        print("")
        print("Please find evaluation results here "+result_file_path)
    return directory_path

  '''
      MAJ : 04112020
      By : Maurras
      Add new function to only execute IForestASD
  '''
  #The goal of this function is to execute the models and show the differents results. 
  #It is the function to call when we want to test differents models 
  #with differents values for parameters
  def run_IForestASD(self, execution_number:int, stream, stream_n_features, window = 100, 
                     estimators = 50, anomaly = 0.5, drift_rate = 0.3, 
                     result_folder="Generated", max_sample=100000, n_wait=200,
                     metrics=['accuracy', 'f1', 'kappa', 'kappa_m', 
                              'running_time','model_size']):
    
    #from skmultiflow.anomaly_detection import HalfSpaceTrees
    from source.iforestasd_scikitmultiflow import IsolationForestStream
    #from source.iforestasd_adwin_scikitmultiflow import IsolationForestADWINStream
    from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    
    
    # Creation f the result csv
    directory_path = 'results/'+str(result_folder)
    self.check_directory(path=directory_path)
    #nb_update
    
    for i in range(execution_number):
        print("*************************************** Execution N° "+str(i)+"**********************************")
        models = [IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "AnomalyRate")
                   ]
        
        result_path = directory_path+'/result_Number'+str(i)+'_for_WS'+str(window)+'_NE'+str(estimators)
        result_file_path = result_path+'.csv'
        # Setup the evaluator
        evaluator = EvaluatePrequential(pretrain_size=1, max_samples=max_sample, 
                                        show_plot=True, 
                                        metrics=metrics, batch_size=1, 
                                        output_file = result_file_path,
                                        n_wait = n_wait) 
        # 4. Run the evaluation 
        evaluator.evaluate(stream=stream, model=models, model_names=['O_IFA'
                                                                    ])
        #Save the stats about models updating
        update_file_path = result_path+"updated_count.csv"
#        update_data = []
#        for model in models:
#            update_data.append(model.model_update)
#        self.array_to_csv(data = update_data, file_path=update_file_path)
        self.models_updated_to_csv(models=models, file_path=update_file_path)
        
        print("")
        print("Please find evaluation results here "+result_file_path)
    return directory_path

  '''
      MAJ : 26112020
      By : Maurras & Mariam
      Add new function to only execute and compare the Four version of IForest ASD
      IForest ASD, SADWIN IFA, PADWIN IFA and NDKSWIN IFA
  '''
  #The goal of this function is to execute the models and show the differents results. 
  #It is the function to call when we want to test differents models 
  #with differents values for parameters
  def run_IForestASDs_comparison(self, execution_number:int, stream, stream_n_features, window = 100, 
                     estimators = 50, anomaly = 0.5, drift_rate = 0.3, 
                     result_folder="Generated", max_sample=100000, n_wait=200,
                     metrics=['accuracy', 'f1', 'kappa', 'kappa_m', 
                              'running_time','model_size'],
                     n_estimators_updated=0.5, updated_randomly=True,
                     alpha=0.01, n_dimensions=1, n_tested_samples=0.01,
                     fixed_checked_dimension = False, fixed_checked_sample=False):
    
    from source.iforestasd_scikitmultiflow import IsolationForestStream
    from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    
    
    # Creation f the result csv
    directory_path = 'results/'+str(result_folder)
    self.check_directory(path=directory_path)
    #nb_update
    
    for i in range(execution_number):
        print("*************************************** Execution N° "+str(i)+"**********************************")
        models = [IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "AnomalyRate",
                              n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly),
                    IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "SADWIN",
                             n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly),
                    IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "PADWIN",
                              n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly),
                    IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "NDKSWIN",
                              n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly,
                              alpha=alpha, n_dimensions=n_dimensions,n_tested_samples=n_tested_samples,
                              fixed_checked_dimension = fixed_checked_dimension,
                              fixed_checked_sample=fixed_checked_sample),
# =============================================================================
#                     IsolationForestStream(window_size=window, n_estimators=estimators, 
#                               anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "PCANDKSWIN",
#                               n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly,
#                               alpha=alpha, n_dimensions=n_dimensions,n_tested_samples=n_tested_samples,
#                               fixed_checked_dimension = fixed_checked_dimension,
#                               fixed_checked_sample=fixed_checked_sample),
# =============================================================================
                   ]
        #models = [
        #            IsolationForestStream(window_size=window, n_estimators=estimators, 
        #                      anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "NDKSWIN",
        #                      n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly,
        #                      alpha=alpha, n_dimensions=n_dimensions,n_tested_samples=n_tested_samples,
        #                      fixed_checked_dimension = fixed_checked_dimension,
        #                      fixed_checked_sample=fixed_checked_sample),
        #           ]
        
        result_path = directory_path+'/result_Number'+str(i)+'_for_WS'+str(window)+'_NE'+str(estimators)
        result_file_path = result_path+'.csv'
        # Setup the evaluator
        evaluator = EvaluatePrequential(pretrain_size=1, max_samples=max_sample, 
                                        show_plot=True, 
                                        metrics=metrics, batch_size=1, 
                                        output_file = result_file_path,
                                        n_wait = n_wait) 
        # 4. Run the evaluation 
        evaluator.evaluate(stream=stream, model=models, model_names=['O_IFA',
                                                                    'SADWIN_IFA',
                                                                    'PADWIN_IFA',
                                                                    'NDKSWIN_IFA'#,
                                                                    #'PCA_NDKSWIN_IFA'
                                                                    ])
        #evaluator.evaluate(stream=stream, model=models, model_names=[
        #                                                            'NDKSWIN_IFA'
        #                                                            ])
        #Save the stats about models updating
        update_file_path = result_path+"updated_count.csv"
#        update_data = []
#        for model in models:
#            update_data.append(model.model_update)
#        self.array_to_csv(data = update_data, file_path=update_file_path)
        self.models_updated_to_csv(models=models, file_path=update_file_path)
        
        print("")
        print("Please find evaluation results here "+result_file_path)
    return directory_path



  '''
      MAJ : 12122020
      By : Maurras
      Add new function to only execute and compare the Four version of IForest ASD
      IForest ASD, SADWIN IFA, PADWIN IFA and NDKSWIN IFA
  '''
  #The goal of this function is to execute the models and show the differents results. 
  #It is the function to call when we want to test differents models 
  #with differents values for parameters
  def run_IForestASDs_comparison2(self, execution_number:int, stream, stream_n_features, window = 100, 
                     estimators = 50, anomaly = 0.5, drift_rate = 0.3, 
                     result_folder="Generated", max_sample=100000, n_wait=200,
                     metrics=['accuracy', 'f1', 'kappa', 'kappa_m', 
                              'running_time','model_size'],
                     n_estimators_updated=0.5, updated_randomly=True,
                     alpha=0.01, n_dimensions=1, n_tested_samples=0.01,
                     fixed_checked_dimension = False, fixed_checked_sample=False):
     
    
    from source.iforestasd_scikitmultiflow_OriginalIFA import IsolationForestStream
    from source.iforestasd_scikitmultiflow_PADWIN import PADWINIsolationForestStream
    from source.iforestasd_scikitmultiflow_SADWIN import SADWINIsolationForestStream
    from source.iforestasd_scikitmultiflow_NDKSWIN import NDKSWINIsolationForestStream
    from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    
    
    # Creation f the result csv
    directory_path = 'results/'+str(result_folder)
    self.check_directory(path=directory_path)
    #nb_update
    
    for i in range(execution_number):
        print("*************************************** Execution N° "+str(i)+"**********************************")
        models = [IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "AnomalyRate",
                              n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly),
                    SADWINIsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "SADWIN",
                             n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly),
                    PADWINIsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "PADWIN",
                              n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly),
                    NDKSWINIsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "NDKSWIN",
                              n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly,
                              alpha=alpha, n_dimensions=n_dimensions,n_tested_samples=n_tested_samples,
                              fixed_checked_dimension = fixed_checked_dimension,
                              fixed_checked_sample=fixed_checked_sample),
                   ]
        result_path = directory_path+'/result_Number'+str(i)+'_for_WS'+str(window)+'_NE'+str(estimators)
        result_file_path = result_path+'.csv'
        # Setup the evaluator
        evaluator = EvaluatePrequential(pretrain_size=1, max_samples=max_sample, 
                                        show_plot=True, 
                                        metrics=metrics, batch_size=1, 
                                        output_file = result_file_path,
                                        n_wait = n_wait) 
        # 4. Run the evaluation 
        evaluator.evaluate(stream=stream, model=models, model_names=['O_IFA',
                                                                    'SADWIN_IFA',
                                                                    'PADWIN_IFA',
                                                                    'NDKSWIN_IFA'#,
                                                                    #'PCA_NDKSWIN_IFA'
                                                                    ])
        #evaluator.evaluate(stream=stream, model=models, model_names=[
        #                                                            'NDKSWIN_IFA'
        #                                                            ])
        #Save the stats about models updating
        update_file_path = result_path+"updated_count.csv"
#        update_data = []
#        for model in models:
#            update_data.append(model.model_update)
#        self.array_to_csv(data = update_data, file_path=update_file_path)
        self.models_updated_to_csv(models=models, file_path=update_file_path)
        
        print("")
        print("Please find evaluation results here "+result_file_path)
    return directory_path

  '''
      MAJ : 04112020
      By : Maurras
      Add new function to only execute the two version of IForestASD : the ADWIN's one and the anomaly rate's one 
  '''
  #The goal of this function is to execute the models and show the differents results. 
  #It is the function to call when we want to test differents models 
  #with differents values for parameters
  def run_IForestASD_comparison(self, execution_number:int, stream, stream_n_features, window = 100, 
                     estimators = 50, anomaly = 0.5, drift_rate = 0.3, 
                     result_folder="Generated", max_sample=100000, n_wait=200,
                     metrics=['accuracy', 'f1', 'kappa', 'kappa_m', 
                              'running_time','model_size'],
                              n_estimators_updated=0.5, updated_randomly=True):
    
    from skmultiflow.anomaly_detection import HalfSpaceTrees
    from source.iforestasd_scikitmultiflow import IsolationForestStream
    from source.iforestasd_adwin_scikitmultiflow import IsolationForestADWINStream
    from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    
    
    # Creation f the result csv
    directory_path = 'results/'+str(result_folder)
    self.check_directory(path=directory_path)
    #nb_update
    
    for i in range(execution_number):
        print("*************************************** Execution N° "+str(i)+"**********************************")
        models = [IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "AnomalyRate",
                              n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly),
                    IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "MAnomalyRate",
                              n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly),
                    IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "SADWIN",
                              n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly),
                    IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "SMADWIN",
                              n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly),
                    IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "PADWIN",
                              n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly),
                    IsolationForestStream(window_size=window, n_estimators=estimators, 
                              anomaly_threshold=anomaly, drift_threshold=drift_rate, version = "PMADWIN",
                              n_estimators_updated=n_estimators_updated, updated_randomly=updated_randomly)#,
                   #HalfSpaceTrees(n_features=stream_n_features, window_size=window, 
                   #              n_estimators=estimators, anomaly_threshold=anomaly)
                   ]
        
        result_path = directory_path+'/result_Number'+str(i)+'_for_WS'+str(window)+'_NE'+str(estimators)
        result_file_path = result_path+'.csv'
        # Setup the evaluator
        evaluator = EvaluatePrequential(pretrain_size=1, max_samples=max_sample, 
                                        show_plot=True, 
                                        metrics=metrics, batch_size=1, 
                                        output_file = result_file_path,
                                        n_wait = n_wait) 
        # 4. Run the evaluation 
        evaluator.evaluate(stream=stream, model=models, model_names=['O_IFA',
                                                                     'M_IFA',
                                                                    'SADWIN_IFA',
                                                                    'MSADWIN_IFA',
                                                                    'PADWIN_IFA',
                                                                    'MPADWIN_IFA'
                                                                    #,'HSTrees'
                                                                    ])
        #Save the stats about models updating
        update_file_path = result_path+"updated_count.csv"
#        update_data = []
#        for model in models:
#            update_data.append(model.model_update)
#        self.array_to_csv(data = update_data, file_path=update_file_path)
        self.models_updated_to_csv(models=models, file_path=update_file_path)
        
        print("")
        print("Please find evaluation results here "+result_file_path)
    return directory_path

  #The goal of this function is to execute the models and show the differents results. 
  #It is the function to call when we want to test differents models 
  #with differents values for parameters
  def run_comparison(self, stream, stream_n_features, window = 100, 
                     estimators = 50, anomaly = 0.5, drift_rate = 0.3, 
                     result_folder="Generated", max_sample=100000, n_wait=200,
                     metrics=['accuracy', 'f1', 'kappa', 'kappa_m', 
                              'running_time','model_size']):
    
    from skmultiflow.anomaly_detection import HalfSpaceTrees
    from source.iforestasd_scikitmultiflow import IsolationForestStream
    from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    
    # Creation f the result csv
    directory_path = 'results/'+str(result_folder)
    self.check_directory(path=directory_path)
    result_file_path = directory_path+'/result_for_WS'+str(window)+'_NE'+str(estimators)+'.csv'
    
    # 2. Prepare for use This function is usefull to have data window by window
    # stream.prepare_for_use() # Deprecated so how to prepare data?
    
    models = [HalfSpaceTrees(n_features=stream_n_features, window_size=window, 
                             n_estimators=estimators, anomaly_threshold=anomaly),
    #IForest ASD use all the window_size for the sample in the training phase
    IsolationForestStream(window_size=window, n_estimators=estimators, 
                          anomaly_threshold=anomaly, drift_threshold=drift_rate)]
    # Setup the evaluator
    evaluator = EvaluatePrequential(pretrain_size=1, max_samples=max_sample, 
                                    show_plot=True, 
                                    metrics=metrics, batch_size=1, 
                                    output_file = result_file_path,
                                    n_wait = n_wait) 
    # 4. Run the evaluation 
    evaluator.evaluate(stream=stream, model=models, model_names=['HSTrees','iForestASD'])
    print("")
    print("Please find evaluation results here "+result_file_path)
    return directory_path
  
  def get_dataset(self, dataset_name="Generator", classification_function=0, 
                  noise_percentage=0.7, random_state=1,
                                      drift_classification_function = 3, drift_random_state = 112,
                                      drift_noise_percentage = 0.0,
                                      drift_start_position = 5000, drift_width = 1000,
                                      n_num_features = 2, n_cat_features = 0):
      #Dataset
      #  Name M(#instances) N(#attributes) Anomaly
      #  Threshold
      #  Http 567498 3 0.39%
      #  Smtp 95156 3 0.03%
      #  ForestCover 286048 10 0.96%
      #  Shuttle 49097 9 7.15%
      if dataset_name=="Generator":
         return self.get_data_generated(classification_function, 
                                        noise_percentage, random_state);
                                        
      elif dataset_name == "DriftStreamGenerator":
          
          return self.get_conceptdrift_data_generated(classification_function = classification_function, 
                                      noise_percentage = noise_percentage, random_state = random_state,
                                      drift_classification_function = drift_classification_function, drift_random_state = drift_random_state,
                                      drift_noise_percentage = drift_noise_percentage,
                                      drift_start_position = drift_start_position, drift_width = drift_width,
                                      n_num_features = n_num_features, n_cat_features = n_cat_features);
      elif dataset_name=="HTTP":
         path = "datasets/HTTP.csv"
         return self.get_file_stream(path);
      elif dataset_name=="ForestCover":
         path = "datasets/ForestCover.csv"
         return self.get_file_stream(path);
      elif dataset_name=="Shuttle":
         path = "datasets/Shuttle.csv"
         return self.get_file_stream(path);
      elif dataset_name=="SMTP":
         path = "datasets/SMTP.csv"
         return self.get_file_stream(path);
      else:
         print("The specified dataset do not exist yet."+ 
               " Try to contact the administrator for any add. "+
               " Or choose between these datasets:['Generator','HTTP','ForestCover','Shuttle','SMTP']");
         return None
  '''
      MAJ : 26112020
      By : Maurras & Mariam
      Add new function to generate stream data from csv file and get it twice
  '''  
  def get_file_stream2(self, path):
      from skmultiflow.data.file_stream import FileStream
      stream = FileStream(path, n_targets=1, target_idx=-1)
      stream2 = FileStream(path, n_targets=1, target_idx=-1)
      stream3 = FileStream(path, n_targets=1, target_idx=-1)
      
      return stream, stream2, stream3
  
  def get_file_stream6(self, path):
      from skmultiflow.data.file_stream import FileStream
      stream = FileStream(path, n_targets=1, target_idx=-1)
      stream2 = FileStream(path, n_targets=1, target_idx=-1)
      stream3 = FileStream(path, n_targets=1, target_idx=-1)
      stream4 = FileStream(path, n_targets=1, target_idx=-1)
      stream5 = FileStream(path, n_targets=1, target_idx=-1)
      stream6 = FileStream(path, n_targets=1, target_idx=-1)
      
      return stream, stream2, stream3, stream4, stream5, stream6
          
  def get_file_stream(self, path):
      from skmultiflow.data.file_stream import FileStream
      return FileStream(path, n_targets=1, target_idx=-1)
  
  def get_data_stream(self, path):
      from skmultiflow.data.data_stream import DataStream
      return
  
    
  def get_data_generated(self,classification_function, noise_percentage, random_state):
      from skmultiflow.data import SEAGenerator
      return SEAGenerator(classification_function=classification_function, 
                          noise_percentage=noise_percentage, random_state=random_state)
  '''
      MAJ : 08112020
      By : Maurras
      Add new function to generate stream data containing anomalies using AnomalySineGenerator
  '''     
  def get_anomalies_data_generated(self, n_samples=10000, n_anomalies=2500, contextual=False,
                 n_contextual=2500, shift=4, noise=0.5, replace=True, random_state=None):
      
      from skmultiflow.data import ConceptDriftStream
      from skmultiflow.data import AnomalySineGenerator
                 
      stream=AnomalySineGenerator(n_samples=n_samples, n_anomalies = n_anomalies, contextual=contextual,
                 n_contextual=n_contextual, shift=shift, noise=noise, replace=replace, random_state=random_state)
      
      #drift_stream=AGRAWALGenerator(classification_function=drift_classification_function, 
      #                        perturbation = drift_noise_percentage, random_state=drift_random_state
      #                        #,n_num_features = n_num_features, n_cat_features = n_cat_features
      #                        )
      
      #return ConceptDriftStream(stream=stream, drift_stream=drift_stream,
      #                          position=drift_start_position, width=drift_width)
      return stream
  
  '''
      MAJ : 05112020
      By : Maurras
      Add new function to generate stream data containing drift using the ConceptDriftStream function
      But this dataset don't contains anomalies. It can contains some noises.
  '''  
  def get_conceptdrift_data_generated(self, classification_function = 0, 
                                      noise_percentage = 0.1, random_state = 112,
                                      drift_classification_function = 3, drift_random_state = 112,
                                      drift_noise_percentage = 0.0,
                                      drift_start_position = 5000, drift_width = 1000,
                                      n_num_features = 2, n_cat_features = 0):
      from skmultiflow.data import ConceptDriftStream
      from skmultiflow.data import AGRAWALGenerator
      
      stream=AGRAWALGenerator(classification_function=classification_function, 
                              perturbation = noise_percentage, random_state=random_state
                              #,n_num_features = n_num_features, n_cat_features = n_cat_features
                              )
      
      drift_stream=AGRAWALGenerator(classification_function=drift_classification_function, 
                              perturbation = drift_noise_percentage, random_state=drift_random_state
                              #,n_num_features = n_num_features, n_cat_features = n_cat_features
                              )
      
      return ConceptDriftStream(stream=stream, drift_stream=drift_stream,
                                position=drift_start_position, width=drift_width)
  '''
      MAJ : 08112020
      By : Maurras
      Add new function to save stream data in .csv file
  '''  
  def save_stream_data_generated(self, stream, result_folder, window = 100, window_number = 100):
      data = self.stream_to_batch(stream=stream, window=window, window_number = 100)
      # Creation f the result csv
      directory_path = 'results/'+str(result_folder)
      self.check_directory(path=directory_path)
      file_path = directory_path+'/'+result_folder+'_dataUsed.csv'
      #dataset = pd.DataFrame(data)
      data.to_csv(file_path, index=None, header=True)
      print("")
      print("Please find the data used on "+file_path)
      return file_path
      
  '''
      MAJ : 08112020
      By : Maurras
      Add new function to generate batch version of the stream in order to save it for future use.
  '''  
  def stream_to_batch(self, stream, window, window_number = 100):
    import pandas as pd
    full_dataset = pd.DataFrame()
    window_used = 0
    while(stream.n_remaining_samples() and window_used < 100):
        #print(str(stream.n_remaining_samples()))
        X, Y =  stream.next_sample(window)
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        dataset = pd.concat([X,Y],ignore_index=True, join='outer', axis=1)
        full_dataset =  pd.concat([full_dataset,dataset],ignore_index=True)
        window_used = window_used + 1
    #print(full_dataset)
    return full_dataset
      
  '''
      MAJ : 21112020
      By : Maurras
      Add new function to save updating result on models for every windows
  '''  
  def array_to_csv(self, data, file_path):
    import pandas as pd
    full_data = pd.DataFrame()
    for d in data:
        X = pd.DataFrame(d)
        full_data = pd.concat([full_data,X],ignore_index=True, join='outer', axis=1)
    full_data.to_csv(file_path, index=None, header=True)
    print("")
    print("Please find the result on "+file_path)
    return file_path
      
  '''
      MAJ : 02122020
      By : Maurras
      Add new function to get the models updated informations (Specific for iforestASD models)
  '''  
  def models_updated_to_csv(self, models, file_path):
    import pandas as pd
    full_data = pd.DataFrame()
    
    for model in models:
        #X = pd.concat([model.model_update,model.model_update_windows],ignore_index=True, join='outer', axis=1)
        full_data = pd.concat([full_data,
                               pd.concat([pd.DataFrame(model.model_update),pd.DataFrame(model.model_update_windows)],
                                         ignore_index=True, join='outer', axis=1)],
          ignore_index=True, join='outer', axis=1)
    full_data.to_csv(file_path, index=None, header=True)
    print("")
    print("Please find the result on "+file_path)
    return file_path
    
  #To transform datasets by replace anomaly label by 1 and normal label by 0
  def prepare_dataset_for_anomaly(self, full_dataset, y_column:int, 
                                  anomaly_label:str='\'Anomaly\'', file_name:str="new"):
      import numpy as np
      import pandas as pd
      
      full_dataset[y_column] = np.where(full_dataset[y_column]==anomaly_label,1,0)
      dataset = pd.DataFrame(full_dataset)
      dataset.drop([0], inplace=True)
      full_file_path = "../datasets/"+file_name+".csv"
      dataset.to_csv(full_file_path, index=None, header=True)
      return dataset
  
  def check_directory(self,path):
      from pathlib import Path
      Path(path).mkdir(parents=True, exist_ok=True) 
      
     
  def merge_file2(self,folder_path, output_file = 'output',skiprows=8):
    import os
    import pandas as pd
    result_update = pd.DataFrame()
    result_exec = pd.DataFrame()
    output_file = folder_path+'/'+output_file
    #print('List of file merged')
    #print()
    no = '.ipynb_checkpoints'
    for file_ in os.listdir(folder_path):
        #print(file_)
        #list.append(file_)
        #print(file_.find("dataUsed"))
        if file_.find("dataUsed") == -1:
            if file_ != no:
                #print(file_)
                
                #df = pd.DataFrame()
                if file_.find("updated_count") != -1:
                    df = pd.read_csv(folder_path+"/"+file_, sep = ',', skiprows=1, header = 0, dtype='unicode', error_bad_lines=False)
                    #df.at[0,'update_'] = df.param.apply(lambda st: st[st.find("WS")+2:st.find("_NE")])[0]
                    df.at[0,'param'] = str(file_)
                    #df.at[0,'ExecNumer'] = df.param.apply(lambda st: str(st)[str(st).find("Number")+6:str(st).find("_for")])[0]
                    df.at[0,'window'] = df.param.apply(lambda st: str(st)[str(st).find("WS")+2:str(st).find("_NE")])[0]
                    df.at[0,'estimators']= df.param.apply(lambda st: str(st)[str(st).find("NE")+2:str(st).find("updated_count.csv")])[0]
                    #df.at[0,'updates']= df.param.apply(lambda st: st[st.find("UP_")+3:st.find(".csv")])[0]
                    #df = pd.concat([df,pd.read_csv(folder_path+"/"+file_, sep = ',', skiprows=1, header = 0, dtype='unicode', error_bad_lines=False)], 
                              #ignore_index=True)
                    if(len(result_update) == 0):
                        result_update = pd.concat([result_update,df], ignore_index=True)
                    else:
                        result_update = pd.concat([result_update,df])
                elif file_.find("_for_WS") != -1:
                    df = pd.read_csv(folder_path+"/"+file_, sep = ',', skiprows=skiprows, header = 0, dtype='unicode', error_bad_lines=False)
                    #df.at[0,'param'] = str(file_)
                    df.at[0,'param'] = str(file_)
                    #df.at[0,'ExecNumer'] = df.param.apply(lambda st: str(st)[str(st).find("Number")+6:str(st).find("_for")])[0]
                    df.at[0,'window'] = df.param.apply(lambda st: str(st)[str(st).find("WS")+2:str(st).find("_NE")])[0]
                    df.at[0,'estimators']= df.param.apply(lambda st: str(st)[str(st).find("NE")+2:str(st).find(".csv")])[0]
                    #df.at[0,'updates']= df.param.apply(lambda st: st[st.find("UP_")+3:st.find(".csv")])[0]
                    if file_.find("_UR") != -1:
                        df.at[0,'update_estimators'] = df.param.apply(lambda st: str(st)[str(st).find("UR")+2:str(st).find("_for_WS")])[0]
                    #df = pd.concat([df,pd.read_csv(folder_path+"/"+file_, sep = ',', skiprows=skiprows, header = 0, dtype='unicode', error_bad_lines=False)], 
                    #          ignore_index=True)
                 #   result = pd.concat([result,df],ignore_index=True)
                    #result_update = pd.concat([result_update,df]).groupby(['AnomalyRate', 'SADWIN', 'PADWIN']).sum()
                    #if(len(result_exec) == 0):
                    result_exec = pd.concat([result_exec,df], ignore_index=True)
                    #else:
                    #    result_exec = pd.concat([result_update,df])
                        #pd.merge(result_update, df, on=[0]).set_index([0]).sum(axis=1)
    #print(result_update)
    #print(result_exec)
    
    result_update.to_csv(output_file+"_updated.csv",index=False)
    result_exec.to_csv(output_file+"_results.csv",index=False)
    
    result_update.to_csv(output_file+"_full.csv",index=False)
    result_exec.to_csv(output_file+"_full.csv", mode='a', header=True)
    
    mycolumns = []
    for idx,column in enumerate(result_exec.columns):
        if idx > len(result_exec.columns)-3:
            mycolumns.append(column)
        elif column.find("mean_") != -1:
            mycolumns.append(column)
    #print(result_exec[mycolumns])
    result_exec[mycolumns].to_csv(output_file+"_usable_results.csv",index=False)
    
    #print(result)
    print("Please find the merged result file on "+str(output_file+"_full.csv"))
    return result_exec, result_update
     
  def merge_file(self, folder_path, output_file = 'output.csv'):
    import os
    import pandas as pd
    result = pd.DataFrame()
    print('List of file merged')
    print()
    no = '.ipynb_checkpoints'
    for file_ in os.listdir(folder_path):
        print(file_)
        #list.append(file_)
        if file_ != no:
            print(file_)
            df = pd.read_csv(folder_path+file_, sep = ',', skiprows=6, header = 0, dtype='unicode', error_bad_lines=False)
            df.at[0,'param'] = str(file_)
            df.at[0,'window'] = df.param.apply(lambda st: st[st.find("WS")+2:st.find("_NE")])[0]
            df.at[0,'estimators']= df.param.apply(lambda st: st[st.find("NE")+2:st.find("_UP")])[0]
            df.at[0,'updates']= df.param.apply(lambda st: st[st.find("UP_")+3:st.find(".csv")])[0]
 
            result = pd.concat([result,df],ignore_index=True)
    #result.sort_values(by = ['window', 'estimators'], inplace= True)
    result.columns=df.columns
    #output_file = 'RESULT_SHUTTLE10K.csv'
    result.to_csv(output_file,index=False)
    
    return result
   
  def data_prep (self, df_forest):
    df_forest.dropna(inplace= True)
    df_forest.sort_values(by = ['window', 'estimators'], inplace= True)
    df_forest.columns = df_forest.columns.str.replace('current_', '')
    df_forest.drop(columns = ['param']).astype(float)
    df_forest=df_forest.drop(columns = ['param']).astype(float)
    df_forest.window = df_forest.window.astype(int)
    df_forest.estimators = df_forest.estimators.astype(int)
    df_forest['Windows_Trees_set_up']='W'+df_forest['window'].astype(str)+'__'+'T'+df_forest['estimators'].astype(str)
    df_forest.columns = df_forest.columns.str.replace('current_', '')
    df_forest.sort_values(by = ['window', 'estimators'], inplace= True)
    
    return df_forest
     
    
      
  '''
      MAJ : 02122020
      By : Maurras
      Add new function to print some graphics after execution
  ''' 
  def print_graphics(self, execution_resuts, updated_results):
        #print(execution_resuts.columns)
        
        return
      
        