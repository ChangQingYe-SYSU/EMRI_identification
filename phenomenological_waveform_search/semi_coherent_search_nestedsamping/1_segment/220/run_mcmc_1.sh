#!/bin/bash
#SBATCH -J emcmc #作业名
#SBATCH -N 1 #调用节点
#SBATCH -n 50 #调用总核心
#SBATCH -o ./emcmc_1.out #标准输休文件
#SBATCH -e ./emcmc_false_1.out #错误输出文件
#SBATCH -p gpu_part # 调用GPU或者CPU
#SBATCH --comment=CALMET

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

source /public/software/anaconda3/bin/activate few_michaelson

python3 /public/home/yecq/para_estimation_Michaelson/fitting_phase/likelihood_function/nestsamping/1_segment/220/nestamping_1_220.py

