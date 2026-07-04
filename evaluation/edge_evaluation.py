
import argparse
import os
import os
import cv2

seism_root = "seism/"


def nms_process(save_dir):
    """
    Do NMS process on edge prediction images
    :param str save_dir: Path of edge prediction directory
    """
    cwd = os.getcwd()
    os.chdir(seism_root)
    new_path = os.path.join("../../", save_dir, "edge")
    abs_path = os.path.abspath(new_path)

    # 替换成你的 nms 目录路径
    nms_dir = os.path.join(abs_path, "nms")
    if os.path.exists(nms_dir):
        files = os.listdir(nms_dir)

        bad_count = 0
        for f in files:
            if f.endswith('.png'):
                path = os.path.join(nms_dir, f)
                img = cv2.imread(path)
                # 如果读出来是空的，说明文件坏了，直接删！
                if img is None:
                    os.remove(path)
                    bad_count += 1
                    print(f"Deleted broken image: {f}")

        print(f"Cleanup done! Found and deleted {bad_count} broken files.")
    # 老师提供的坚不可摧的 Singularity 启动头
    # matlab_base_cmd = "/opt/app/singularity/bin/singularity exec --bind /usr/lib64/libXt.so.6:/usr/lib64/libXt.so.6 --bind /usr/lib64/libICE.so.6:/usr/lib64/libICE.so.6 --bind /opt/app/MATLAB_2024b_Free:/opt/app/MATLAB_2024b_Free /opt/app/sif/rockylinux9.sif /opt/app/MATLAB_2024b_Free/bin/matlab"

    # 拼接上你的执行逻辑 (注意这里我把老师给的 -nodisplay -nosplash 补全了我们之前的无UI参数)
    # cmd_eval = f"{matlab_base_cmd} -nodisplay -nosplash -nodesktop -singleCompThread -r \"nms_process('%s');exit\"" % abs_path"
    # os.system(cmd_eval)
    os.system("/opt/app/singularity/bin/singularity exec --bind /usr/lib64/libXt.so.6:/usr/lib64/libXt.so.6 --bind /usr/lib64/libICE.so.6:/usr/lib64/libICE.so.6 --bind /opt/app/MATLAB_2024b_Free:/opt/app/MATLAB_2024b_Free /opt/app/sif/rockylinux9.sif /opt/app/MATLAB_2024b_Free/bin/matlab -nodisplay -nosplash -nodesktop -r \"nms_process('%s');exit\"" % abs_path)
    os.chdir(cwd)


def eval_edge_predictions(dataset, exp_name, save_dir):
    """
    Evaluate edge predictions using seism in MatLab
    :param str dataset: Dataset name
    :param str exp_name: Name of experiment
    :param str save_dir: Path of edge prediction directory
    """
    print("Evaluate edge predictions using seism in Matlab.")

    print("Generate MATLAB script for evaluation...")
    # Generate MATLAB script
    script_base = os.path.join(seism_root, "pr_curves_base.m")
    with open(script_base) as f:
        seism_file = f.readlines()
    seism_file = [line.rstrip() for line in seism_file]
    output_file = seism_file[0:1]
    output_file += ["database = '%s';" % dataset]
    output_file += ["% 限制并行池大小以避免许可证问题"]
    output_file += ["p = gcp('nocreate'); if ~isempty(p), delete(p); end"]
    output_file += ["c = parcluster('local'); c.NumWorkers = 48; parpool(c, c.NumWorkers);"]


    output_file += seism_file[1:12]

    # Add method
    print("Add method: %s" % exp_name)
    output_file += ["methods(end+1).name = '%s';" % (exp_name)]
    real_nms_path = os.path.abspath(os.path.join(save_dir, "edge", "nms"))
    real_nms_path = real_nms_path.replace("/evaluation/task_predictions", "/task_predictions")
    print("Real NMS path: %s" % real_nms_path)
    output_file += ["methods(end).dir = '%s';" % real_nms_path]
    # output_file += ["methods(end).dir = '%s';" % os.path.join("../", save_dir, "edge", "nms")]
    output_file.extend(seism_file[13:49])

    # Add path to save output
    print("Add output path: %s" % os.path.join(save_dir, "edge_test.txt"))
    real_output_txt = os.path.abspath(os.path.join(save_dir, "edge_test.txt"))
    real_output_txt = real_output_txt.replace("/evaluation/task_predictions", "/task_predictions")
    print("Real output txt path: %s" % real_output_txt)
    output_file += ["\t\t\tfilename = '%s';" % real_output_txt]
    # output_file += ["\t\t\tfilename = '%s';" % (os.path.join("../", save_dir, "edge_test.txt"))]
    output_file += seism_file[50:]

    # Save script file
    print("Save MATLAB script file to seism directory.")
    output_file_path = os.path.join(seism_root, exp_name + ".m")
    with open(output_file_path, "w") as f:
        for line in output_file:
            f.write(line + "\n")

    # Go to seism directory and perform evaluation
    print("Go to seism directory and run evaluation. Please wait...")
    cwd = os.getcwd()
    os.chdir(seism_root)
    # 老师提供的坚不可摧的 Singularity 启动头
    matlab_base_cmd = "/opt/app/singularity/bin/singularity exec --bind /usr/lib64/libXt.so.6:/usr/lib64/libXt.so.6 --bind /usr/lib64/libICE.so.6:/usr/lib64/libICE.so.6 --bind /opt/app/MATLAB_2024b_Free:/opt/app/MATLAB_2024b_Free /opt/app/sif/rockylinux9.sif /opt/app/MATLAB_2024b_Free/bin/matlab"

    # 拼接上你的执行逻辑 (注意这里我把老师给的 -nodisplay -nosplash 补全了我们之前的无UI参数)
    cmd_eval = f"{matlab_base_cmd} -nodisplay -nosplash -nodesktop -nojvm -r \"ps=parallel.Settings; ps.Pool.AutoCreate=false; {exp_name};exit\""
    os.system(cmd_eval)
    # os.system('matlab -nodisplay -nosplash -nodesktop -singleCompThread -r "%s;exit"' % (exp_name))
    os.chdir(cwd)


def display_edge_eval_result(exp_name, save_dir):
    """
    Display edge evaluation result and clean up files
    :param str exp_name: Name of experiment
    :param str save_dir: Path of edge prediction directory and evaluation result
    """
    # Collect results from txt file
    print("Add output path: %s" % os.path.join(save_dir, "edge_test.txt"))
    real_output_txt = os.path.abspath(os.path.join(save_dir, "edge_test.txt"))
    real_output_txt = real_output_txt.replace("/evaluation/task_predictions", "/task_predictions")
    with open(os.path.join(real_output_txt), "r") as f:
        seism_result = [line.strip() for line in f.readlines()]

    eval_dict = {}
    for line in seism_result:
        metric, score = line.split(":")
        eval_dict[metric] = float(score)

    # Print result
    print("Edge Detection odsF: %.4f" % (100 * eval_dict["odsF"]))

    # Cleanup - Important. Else Matlab will reuse the files.
    print("Cleanup result files in seism.")

    result_path = os.path.join(seism_root, "results/%s/" % exp_name)
    os.system("rm -rf %s" % result_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", required=True)
    parser.add_argument("--results_dir", type=str, default="../results", help="directory of results")
    parser.add_argument("--dataset", type=str, default="PASCALContext", help="PASCALContext or NYUD")
    parser.add_argument("--nms", action="store_true", help="Whether to do NMS.")
    parser.add_argument("--done", action="store_true", help="Whether evaluation has been done.")
    args = parser.parse_args()

    # get save directory
    results_dir = args.results_dir
    for exp_name in args.exp:
        save_dir = os.path.join(results_dir, exp_name, "predictions")
        if args.nms:
            nms_process(save_dir)
        eval_edge_predictions(args.dataset, exp_name, save_dir)
        display_edge_eval_result(exp_name, save_dir)
        # if not args.done:
        #     # Step1: NMS process and Evaluate edge predictions using seism in Matlab
        #     if args.nms:
        #         nms_process(save_dir)
        #     eval_edge_predictions(args.dataset, exp_name, save_dir)
        #     display_edge_eval_result(exp_name, save_dir)
        # else:
        #     # Step2: If evaluation is done, display result
        #     display_edge_eval_result(exp_name, save_dir)
