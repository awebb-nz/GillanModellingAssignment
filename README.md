
# Task

Perform Maximum Likelihood Estimation on the formal model and a subset of the data from Gillan et al., (2016).

## Running the code

In order to run the code, clone this repository, change into the root directory, and install the required libraries listed in `requirements.txt`/`Pipfile`. For example to create anew virtual environment for this project using pipenv, do:
```sh
pipenv install
```

However you choose to install the libraries, you must then **activate the environment** and run:
```sh
install_cmdstan
```

When the required libraries are installed and setup, the model itself can be run as follows within python:
```python
from main import *

# This will generate the raw data files from running the model
generate_fit_dataframe("./output", stan_file="gillan_model.stan")
# This will generate a single csv file which contains all of the 
# weights per subject
summarise_datafile("./output")
```
or from the command line:
```sh
python main.py output gillan_model.stan
```

## Viewing the result files

First, please note that (as discussed in the Results section below) I do not believe these results are correct. That said, the results are contained in three csv files in the `output` directory:
- `gillan_model.csv` contains the raw output from running the model with cmdstanpy
- `subjects.csv` gives a mapping from subject id to number in the raw output
- `collated.csv` contains the collated output from the previous two files, with rows being subjects, and columns being the weights associated with each subject.

# Data

Data was provided in a [gitlab repository](https://gitlab.tuebingen.mpg.de/cgagne/gillan_model_assignment), along with a README which explained the meaning of the data columns.

It should be noted that, by looking at the activity associated with the owner of the repository ([here](https://gitlab.tuebingen.mpg.de/cgagne)), it is possible to get access to old versions of the repository which contain pre-calculated fits.

# Model 

The formal model to be fit was specified in [Gillan et al., 2016](https://elifesciences.org/articles/11305). The model itself is based on earlier studies by Otto et al., 2013 and Daw et al., 2011.

# Process

Because I have little experience at running this sort of formal model, the process I followed involved:
- Background reading into running formal models. This also included evaluating the different software libraries available
- Loading and processing the data files, so that they could be used as input to the model
- Specifying and running the actual model, using the data given

## Background reading

Initially, this involved reading Gillan et al., 2016, as well as Daw et al, 2011 and Otto et al., 2013, to get an understanding of the particulars of the experiment, and to get an idea of how the model was defined. The model specified in Gillan et al, involved six free parameters per subject: five weights (beta_MB, beta_MF0, beta_MF1, beta_Sticky, and beta_Q_Stage2, where MB is the model-based component and MF is the model-free component) and a learning rate (alpha). This differed from the earlier papers, which (as I understand it) had a single weight to represent the model-free component.

Because I am most familiar with python for data analysis, I looked at what libraries were available to specify and optimise this kind of model. The most common libraries seemed to be [pystan](https://github.com/stan-dev/pystan)/[cmdstanpy](https://github.com/stan-dev/cmdstanpy) (which are both python frontends to the stan modelling library) and [PyMC3](https://docs.pymc.io/). Because I have seen stan before (although I haven't used pystan/cmdstanpy), and because cmdstanpy seemed easier to install than pystan I proceeded with cmdstanpy.

Therefore, I read around as much as possible for suggestions and tips on writing stan models, calculating maximum likelihood for such a model, and using stan from python. As part of this process, I also found a repository (on the OSF site) for a version of the Daw et al, 2011 model, from a later study by Claire Gillan ([Gillan et al., 2021](https://www.cambridge.org/core/journals/psychological-medicine/article/experimentally-induced-and-realworld-anxiety-have-no-demonstrable-effect-on-goaldirected-behaviour/3A1BB3C05B8B92A52764BB6468CC3193)), which provided a [definition of that model](https://osf.io/w4yfp/) in stan.
## Datafile loading

The next step was to load the data in python, and to perform whatever transformations were necessary to turn the data into the input for a model. 

The data was in individual per-participant csv file, each of which had some preamble, and then the actual trial data. For each file, the preamble was skipped, and then the remaining rows were loaded into a [pandas](https://pandas.pydata.org/) dataframe. 

From the model specification, it seems that only four of the 15 columns are necessary for the model:
- the stage 1 response (F)
- the stage 2 response (J)
- the stage 2 state (L)
- the reward (N)

The response columns had to be converted from left/right into 0/1 for use in the model, and likewise the state and reward columns had to be cast from strings to ints.

Those columns were collected, for each subject, into a list of [numpy](https://numpy.org/) ndarrays, which is what cmdstanpy is expecting as input. I also wrote code to save the raw output from cmdstanpy to csv, and to collate the resulting csv files into a single result file. This code can be seen in `main.py` in `generate_fit_dataframe` and `summarise_datafile` respectively.

## Model specification

The model was implemented according to the specification in Gillan et al, and is given in `gillan_model.stan`. The inputs to the model are as follows:
```stan

    int num_subj; // The number of subjects
    int num_trials; // The number of trials per subject
    int s1_responses[num_subj, num_trials]; // Stage 1 choices
    int s2_responses[num_subj, num_trials]; // Stage 2 choices
    int s2_state[num_subj, num_trials]; // Stage 2 states
    int rewards[num_subj, num_trials]; // Trials rewarded
```
and the per-participant weights are:
```stan

    real<lower=0,upper=1> alphas[num_subj];
    real beta_mb[num_subj];
    real beta_mf0[num_subj];
    real beta_mf1[num_subj];
    real beta_stick[num_subj];
    real beta_s2[num_subj];
```

In order to update the weights, a number of values needed to be calculated on a per-participant or per-trial basis. For example, the following were directly involved in weight update calculations:
```stan
    real qt_stage2[2,2]; 
    real q_mb[2];
    real q_mf_0[2];
    real q_mf_1[2];
    int changed;
```
while the following was used to count the number of state transitions based on stage 1 responses:
```stan
    int state_transitions[2,2];
```

There was also some transformations that needed to be performed to change from responses (which were 0 or 1) and states (which were 2 or 3) to 1-based indexing as used in stan arrays. In particular, because it seemed unclear, rewards were also adjusted (from 0 or 1 to -1 or 1), to better capture the effects of a negative reward.

The rest of the model was an implementation of Gillan et al, informed by the model specification from Gillan et al., 2021, mentioned above. Most particularly, the priors used in my model definition were (as noted in the model itself) taken from the later study, because I didn't know, from my own reading, what priors to use.
# Results and discussion

The first thing to note is that the results are different from those found in the old versions of the task repository. This, of course, suggests that my results are incorrect.

There are a number of possible reasons as to why my results may be wrongs. Firstly, and most obviously, I may have made a mistake in the model (or in my understanding of the problem). This is the most likely reason. 

Another possible explanation is that, due to the limitations of my home computing setup, I am running the `optimize` method of the model (see [specification](https://mc-stan.org/cmdstanpy/api.html#cmdstanpy.CmdStanModel.optimize)), rather than using the `sample` method (see [specification](https://mc-stan.org/cmdstanpy/api.html#cmdstanpy.CmdStanModel.sample)) as was done in Gillan et al., 2021. The reason for this is that my machine didn't have the memory required to run the latter method to completion.

I am not sure whether the memory usage was an effect of the model as I have defined it, or whether it was due to the python wrapper (although, monitoring memory usage suggests it was partially the latter). In the case that it was caused by the model as I defined it, I imagine memory usage could be improved by:
- utilising vectorisation
- running each subject separately

In the first case, the explicit loops over arrays used in the model could be replaced by vectors and matrices which are better able to be optimised by the stan compiler. This would, however necessitate precomputing the per-trial variables (such as `state_transitions`). In the second case, intermediate result files would have to be written to allow sharing of any required variable, while also removing the need to store all of the data in memory at the same time.