data {
    int num_subj; // The number of subjects
    int num_trials; // The number of trials per subject
    int s1_responses[num_subj, num_trials]; // Stage 1 choices
    int s2_responses[num_subj, num_trials]; // Stage 2 choices
    int s2_state[num_subj, num_trials]; // Stage 2 states
    int rewards[num_subj, num_trials]; // Trials rewarded

}
parameters {
    // The six per-subject parameters
    real<lower=0,upper=1> alphas[num_subj];
    real beta_mb[num_subj];
    real beta_mf0[num_subj];
    real beta_mf1[num_subj];
    real beta_stick[num_subj];
    real beta_s2[num_subj];

    // These values are taken from Gillan et al., 2021
    real<lower=0,upper=1> a1m;
    real<lower=0> a1ss;
    real b1mm;
    real<lower=0> b1ms;
    real b1tm;
    real<lower=0> b1ts;
    real b2tm;
    real<lower=0> b2ts;
    real b2m;
    real<lower=0> b2s;
    real bcm;
    real<lower=0> bcs;
}
// These values are taken from Gillan et al., 2021
transformed parameters {
  real a1a;
  real a1b;
  a1a = a1m * pow(a1ss,-2);
  a1b = pow(a1ss,-2) - a1a;
}
model {

    // These values are taken from Gillan et al., 2021
    bcm ~ normal(0,100);
    bcs ~ cauchy(0,2.5);
    b1mm ~ normal(0,100);
    b1ms ~ cauchy(0,2.5);
    b1tm ~ normal(0,100);
    b1ts ~ cauchy(0,2.5);
    b2tm ~ normal(0,100);
    b2ts ~ cauchy(0,2.5);
    b2m ~ normal(0,100);
    b2s ~ cauchy(0,2.5);

    for (subj in 1:num_subj){
        // These values are taken from Gillan et al., 2021
        alphas[subj] ~ beta(a1a,a1b);
        beta_mb[subj] ~ normal(b1mm,b1ms);
        beta_mf0[subj] ~ normal(b1tm,b1ts);
        beta_mf1[subj] ~ normal(b2tm,b2ts);
        beta_s2[subj] ~ normal(b2m,b2s);
        beta_stick[subj] ~ normal(bcm,bcs);

        real qt_stage2[2,2] = {{0, 0}, {0, 0}}; // q value for stage2, for state * response
        real q_mb[2] = {0, 0}; // model-based Q
        real q_mf_0[2] = {0, 0}; // TD(0)
        real q_mf_1[2] = {0, 0}; // TD(1)
        int state_transitions[2,2] = {{0, 0}, {0, 0}}; // counting state transitions from response1 to state
        int previous_response1 = 0; // We need to know whether the response changed from last time
        int changed = 0;
        
        //These are for changing between input values and array
        //indices
        int response1_ind = 0;
        int response2_ind = 0;
        int state = 0;
        int reward = 0;
        int non_response1_ind = 0;
        int non_response2_ind = 0;
        int non_state = 0;

        for (trial in 1:num_trials){
            //responses are 0,1 but array indices are 1,2
            response1_ind = s1_responses[subj, trial] + 1;
            response2_ind = s2_responses[subj, trial] + 1;
            //states are 2,3 but indices are 1,2
            state = s2_state[subj, trial] - 1;
            //I think that there should be a negative reward, in the case 
            //there is no reward
            reward = (rewards[subj,  trial] ? 1: -1);
            //needed to calculate stickiness
            changed = (s1_responses[subj, trial] == previous_response1 ? 1: 0);

            if (state_transitions[1, 1] >= state_transitions[1, 2]) {
                //this means that state 1 is the more common outcome of response 1
                //so, max of state 1
                q_mb[1] = fmax(qt_stage2[1, 1], qt_stage2[1, 2]);
            } else {
                q_mb[1] = fmax(qt_stage2[2, 1], qt_stage2[2, 2]);
            }
        
            if (state_transitions[2, 1] >= state_transitions[2, 1]) {
                //this means that state 2 is the more common outcome of response 2
                //so, max of state 2
                q_mb[2] = fmax(qt_stage2[2, 1], qt_stage2[2, 2]);
            } else {
                q_mb[2] = fmax(qt_stage2[1, 1], qt_stage2[1, 2]);
            }

            //update beta_s2
            s2_responses[subj, trial] ~ bernoulli_logit((beta_s2[subj]) * (qt_stage2[state, 2] - qt_stage2[state, 1]));
            //update the other weights
            s1_responses[subj, trial] ~ bernoulli_logit(
                ((beta_mb[subj]) * (q_mb[2] - q_mb[1]))
                + ((beta_mf0[subj]) * (q_mf_0[2] - q_mf_0[1]))
                + ((beta_mf1[subj]) * (q_mf_1[2] - q_mf_1[1]))
                + ((beta_stick[subj]) * changed)
            );
            
            q_mf_0[response1_ind] = q_mf_0[response1_ind] * (1 - alphas[subj]) + qt_stage2[state, response2_ind];
            q_mf_1[response1_ind] = q_mf_1[response1_ind] * (1 - alphas[subj]) + reward;

            //update the trial-by-trial state variables
            state_transitions[response1_ind, state] = state_transitions[response1_ind, state] + 1;
            qt_stage2[state, response2_ind] = qt_stage2[state,response2_ind] * (1 - alphas[subj]) + reward;
            previous_response1 = s1_responses[subj, trial];

            //decay unchosen actions and unvisited states
            non_response1_ind = (response1_ind == 1 ? 2: 1);
            non_response2_ind = (response2_ind == 1 ? 2: 1);
            non_state = (state == 1 ? 2: 1);
            qt_stage2[non_state, 1] = qt_stage2[non_state, 1] * (1 - alphas[subj]);
            qt_stage2[non_state, 2] = qt_stage2[non_state, 2] * (1 - alphas[subj]);
            qt_stage2[state, non_response2_ind] = qt_stage2[state, non_response2_ind] * (1 - alphas[subj]);
            q_mf_0[non_response1_ind] = q_mf_0[non_response1_ind] * (1 - alphas[subj]);
            q_mf_1[non_response1_ind] = q_mf_1[non_response1_ind] * (1 - alphas[subj]);
        }
    }
}