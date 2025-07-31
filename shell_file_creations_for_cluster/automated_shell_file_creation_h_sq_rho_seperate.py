# create shell scripts for running the python files if they are not already created
# append the sbatch commands to a .txt file
import os
if os.path.exists('shell_scripts'):
     # remove the directory and its contents
    for file in os.listdir('shell_scripts'):
        os.remove(f'shell_scripts/{file}')
else:
    os.makedirs('shell_scripts')


for ddp_str in ['eur', 'afr', 'amr', 'sas', 'eas']:
    for h_sq in [0.25, 0.5]:
        for rho in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
                with open("shell_scripts/temp.sh", "w+") as f:
                    f.write("#!/bin/bash\n")
                    f.write(f"#SBATCH --job-name=neural_network_exisitng_{ddp_str}_batch_norm_h_sq_{h_sq}_rho_{rho}\n")
                    f.write("#SBATCH --mail-user=skumar26@uthsc.edu\n")
                    f.write("#SBATCH --mail-type=BEGIN,END,FAIL\n")
                    f.write("#SBATCH --account=ACF-UTHSC0001\n")
                    f.write("#SBATCH --partition=campus\n")
                    f.write("#SBATCH --qos=campus\n")
                    f.write(f"#SBATCH --output=DDP_output/{ddp_str}_h_sq_{h_sq}_o_files/neural_network_exisitng.o%j\n")
                    f.write(f"#SBATCH --error=DDP_error/{ddp_str}_h_sq_{h_sq}_e_files/neural_network_exisitng.e%j\n")
                    f.write("#SBATCH --nodes=1\n")
                    f.write("#SBATCH --ntasks-per-node=8\n")
                    f.write("#SBATCH --time=01-00:00:00\n")
                    f.write("###########################################\n")
                    # f.write("cd /nfs/home/skumar26/synthetic_data_generation_regression\n")
                    f.write("module purge\n")
                    f.write("module load gcc/10.2.0\n")
                    f.write("module load Python/3.9.10-gcc\n")
                    f.write("source /nfs/home/skumar26/GPU_lower_env/tfgpu/bin/activate\n")
                    f.write(f"python neural_network_exisitng_Layernorm.py '{ddp_str}' {h_sq} {rho}\n")
                    f.write("echo 'The execution has been done.'\n")
                    f.close()
                with open("shell_scripts/temp.sh", "rb") as fr, open(f"shell_scripts/neural_network_exisitng_{ddp_str}_batch_norm_h_sq_{h_sq}_rho_{rho}.sh", "wb") as fw:
                    for l in fr.readlines():
                        fw.write(l.replace(b"\r\n",b"\n"))
                    f.close()

                # append a line to a .txt file and create if it doesn't exist
                with open("shell_scripts/sbatch_commands_scripts.txt", "a+") as f:
                    f.write(f"sbatch neural_network_exisitng_{ddp_str}_batch_norm_h_sq_{h_sq}_rho_{rho}.sh\n")
                    f.close()    
