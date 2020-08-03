# active_learning_for_cbeta_accelerator
Apply active learning for more efficient data sampling. 

We want to train a random forest to represent the cbeta accelerator. 

The training data comes from physics-based simulation. 
However, physics-based simulation is really expensive to run. Therefore, we don't want to generate those simulations based on random inputs. 
We apply the idea of active learning for selecting what kind of samples to generate. We want to generate data where the current random forest is doing the worst. 
