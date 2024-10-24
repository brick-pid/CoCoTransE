#! /bin/bash

if [ "$#" -ne 3 ]; then
    echo "用法: $0 [源代码文件] [目标可执行文件][代码序号]"
    exit 1
fi

source /home/sjw/ljb/cangjie/envsetup.sh 
# 提取参数
source_file="$1"
output_file="$2"
folder_name="$3"

# 编译源代码文件，删除颜色代码
cjc "${folder_name}/${source_file}" -o "${folder_name}/${output_file}" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "${folder_name}/error.json"

# 检查编译是否成功，若成功，则删除编译报错信息文件
if [ -f "${folder_name}/${output_file}" ]; then
    #echo "编译成功!"
    ./"${folder_name}/${output_file}" > "${folder_name}/output.json"
    rm -f "${folder_name}/error.json"
    rm -f "${folder_name}/default.bchir2"
    rm -f "${folder_name}/default.cjo"
    rm -f "${folder_name}/${output_file}"
    exit 0
else
    #echo "编译失败!请查看错误信息。"
    exit 1
fi
