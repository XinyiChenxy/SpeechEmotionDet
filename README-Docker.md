# SER GPU 开发环境

```bash
cd /home/xinyic/SER
./build.sh
./run.sh
```

宿主机 `/home/xinyic/SER` 挂载到容器 `/workspace`，默认工作目录为 `/workspace`。
启动参数包含 `--gpus all -it --rm --shm-size=8g`，进入 bash。
容器名为 `ser-workspace`。在另一个宿主机终端进入同一容器：

```bash
docker exec -it -w /workspace ser-workspace bash
```

`./verify.sh` 用非交互临时容器检查依赖、GPU 运算、工作目录和挂载写入。
run.sh 按宿主机执行用户 UID/GID 运行，缓存保存在宿主机项目 `.cache`。
退出主 bash 后容器自动删除，项目和缓存保留。再次执行 run.sh 即可新建容器。

基础镜像为 PyTorch 2.0.1 / CUDA 11.7 / cuDNN 8 devel（Python 3.10），兼容已检查的 NVIDIA 535.183.01 驱动和 RTX 3090 Ti。
依赖沿用此前审阅的 Vesper 官方源码 API 所选组合，NumPy 1.23.5 保留旧 API。
安装 ffmpeg、libsndfile、git、git-lfs 及编译工具；不复制项目、数据或权重进镜像。
build.sh 透传宿主机已有代理环境变量，不在文件中保存代理凭据。

此前项目目录已按要求删除；此目录为新建环境目录，不包含原数据集或训练代码。

镜像已创建 xinyic 用户及组；build.sh 按执行用户 UID/GID 构建，默认以 xinyic 运行，避免交互终端出现 `I have no name!`。

run.sh 也透传宿主机代理环境变量，供容器内 Codex 等程序访问网络。此前容器没有代理环境变量，直连 auth.openai.com 被重置；代理测试可建立 TLS，但首页返回 HTTP 403，尚未验证实际登录成功。需在宿主机已有可用代理环境的终端启动 run.sh，再在容器扩展中重新登录。
