condor_rm hhao9
rm -rf conjugate_priors/logs/*
# rm -rf soft_kmeans/logs/*
# rm -rf img/**/*
# rm -rf temp_json_results/*
condor_submit /home/hhao9/ebm_package/eval_cp.sub
# condor_submit /home/hhao9/ebm_package/eval_sk.sub