import time
import datetime
import os
import shutil


unix_time_request = time.time()
time_request = datetime.datetime.fromtimestamp(unix_time_request)
unix_time_request = str(unix_time_request)
time_request_str = str(time_request).split('.')[0]
time_request_str = time_request_str.replace(' ','-')

"""
Begin Edit Parameters
"""

nGPUs = 1
account = 'cmcdevitt-roy'
qos = 'cmcdevitt-roy'
mem_per_gpu = '80gb'
allowed_runtime = '120:00:00' #hours:minutes:seconds

parameter_list = []

#Enter input parameters below

parameter_list.append("--Ra=1e5")           #starts at sys.argv[3]
parameter_list.append("--nepochs=5e4")      #sys.argv[4]


#Enter the input parameters above

"""Don't touch this section"""
job_specs = ""
for ii in parameter_list:
    job_specs+=(ii)
job_spec_input = job_specs.replace('--'," --")

"""
End Edit Parameters
"""

job_name = time_request_str[11:]

folder_name = 'models/'+time_request_str+job_specs+'/'

os.mkdir(folder_name)
os.mkdir(folder_name+'data')
os.mkdir(folder_name+'Big_Plots')
#os.mkdir(folder_name+'Ghia_Check')
os.mkdir(folder_name+'Individual_Plots')
os.mkdir(folder_name+'Loss_Plots')


shutil.copyfile('og_Func_Fac.py',folder_name+'/Func_Fac.py')
shutil.copyfile('og_Functions.py',folder_name+'/Functions.py')
shutil.copyfile('og_Script.py',folder_name+'/Script.py')
shutil.copyfile('og_Script_Generator.py',folder_name+'/Script_Generator.py')

file_location = folder_name+'Script.py'



with open(folder_name+'job.slurm','w') as f:
    f.write('#!/bin/bash')
    f.write('\n#SBATCH --job-name='+job_name)
    f.write('\n#SBATCH -p')
    f.write('\n#SBATCH gpu')
    f.write('\n#SBATCH --gpus=a100:'+str(nGPUs))
    f.write('\n#SBATCH --account='+account)
    f.write('\n#SBATCH --qos='+qos)
    f.write('\n#SBATCH --mem-per-gpu='+mem_per_gpu)
    f.write('\n#SBATCH --time='+allowed_runtime)
    f.write('\n')
    
    f.write('\nmodule load cuda/11.1.0')
    f.write('\nmodule load conda')
    f.write('\n conda activate env/')
    
    code_start_time_unix = time.time()
    code_start_time_str = str(datetime.datetime.fromtimestamp(code_start_time_unix))
    code_start_time_unix = str(code_start_time_unix)

    f.write('\nnvidia-smi > '+folder_name+'nvidia-output.txt')
    f.write('\n')

    f.write("\necho ''")
    f.write("\necho -e 'Request Start Time:\t'"+time_request_str)
    f.write("\nCODE_START_TIME_UNIX=`date +\%s`")
    f.write('\nCODE_START_TIME_DATE=$(date "+%F-%H:%M:%S")')
    f.write("\necho -e 'Code Start Time:   \t'$CODE_START_TIME_DATE")

    #f.write("\necho -e 'Code Start Time:   \t'"+code_start_time_str)
    f.write("\necho ''")
    f.write('\n')

    f.write('\npython '+" "+file_location+" "+folder_name+" "+"$CODE_START_TIME_UNIX"+job_spec_input)


    f.write('\n')
    f.write("\necho -e 'Request Start Time:\t'"+time_request_str)
    f.write("\necho -e 'Code Start Time:   \t'$CODE_START_TIME_DATE")
    f.write('\necho -e "Code End Time:     \t"$(date "+%F-%H:%M:%S")')
    f.write("\necho ''")

os.system('dos2unix -q '+folder_name+'job.slurm')

with open(folder_name+'send.sh','w') as f:
    f.write('sbatch --output='+folder_name+'log.out --export=DATE_TIME='+time_request_str+' '+folder_name+'job.slurm')
os.system('dos2unix -q '+folder_name+'send.sh')
os.system('bash '+folder_name+'send.sh')
