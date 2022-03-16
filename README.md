# master-thesis



## Structure

When using this project a directory called:
FBR will be create in your home directory this will get the following structure:
- FBR/
    - NMT/
        - model_name_1/
            - data/
                - sampled data
            - model 
        - model_name_2
        - model_name_3
    - PREDICTIVE/
        - preprocessed / 
            - model_name_1/
                - util_function_1/
                    -  features_1/
                        - preprocessed_data
                    -  features_2
                - util_function_2/
        - trained_models
            - predictive_model_1
            - predictive_model_2
            - ....
            
            
In the NMT map the NMT models can be stored together with the data they have generated

In the predictive map we put the preprocessed data for reuse between runs.
Furthermore we store the trained models there. 


            