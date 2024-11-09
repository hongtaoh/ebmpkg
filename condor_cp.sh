condor_rm hhao9
rm -rf conjugate_priors/logs/*
rm -rf conjugate_priors/img/*
rm -rf conjugate_priors/temp_json_results/*
condor_submit /home/hhao9/ebm_package/eval_cp.sub