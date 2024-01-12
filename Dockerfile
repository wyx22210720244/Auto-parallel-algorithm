FROM reg.docker.alibaba-inc.com/dadiprod/dadi_zhouhb_tf_on_hippo:aop_torch_online_llm_for_torch210_ppu_20231211_2345  as builder

# 安装编译依赖

WORKDIR /root
#COPY /nccl-tests  /root/nccl-tests
#RUN #yum  install git tmux htop tar -y
WORKDIR /root/nccl-tests
#RUN  make -j64