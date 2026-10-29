# SuperMimic 部署源码快照

本机实际部署目录 `rl_controller/` 是独立的内网 Git 仓库，并被主仓库忽略。
此目录保存本项目的 SuperMimic 控制器、配置、构建入口及测试，便于随主项目在
GitHub 上审阅和备份。`source_snapshot.json` 记录每个源文件的 SHA256。

快照包含 SDK 2.2.37 位置 PD 适配、X 松键按菜单顺序切换、连续策略参考桥接、
完整动作播放和 END_HOLD 末帧策略保持。当前配置使用 80000 次更新模型，默认
`walk1_subject2`。模型权重和动作二进制需另行导出，不在此目录中。

公司 SDK、AMP 源码、Git 历史、上传脚本及凭据、第三方二进制、构建产物和动作
数据均不包含在快照内。兼容消息定义只供本机隔离测试；真机以 SDK 2.2.37
提供的消息及服务为准。

在已有且匹配的 SDK 部署工程上使用时，先备份该工程，再将此目录的源码覆盖到
对应 `rl_controller/` 工作目录，准备 ONNX Runtime 1.19、ROS 2 Humble、Eigen3、
yaml-cpp 和导出的模型动作包。它不是完整公司 SDK 仓库或开箱即用的二进制安装包。
本机的构建、导出工具仍使用项目根目录下原来的 `rl_controller/`。

维护快照：修改本机实际代码并验证后，在项目根目录执行：

```bash
python3 scripts/snapshot_supermimic_deployment.py
```

本机配置与状态说明见 [SUPERMIMIC_DEPLOYMENT.md](SUPERMIMIC_DEPLOYMENT.md)，
训练检查点到部署包的流程见 [CASBOT02_DEPLOY_EXPORT.md](../../docs/CASBOT02_DEPLOY_EXPORT.md)。
测试通过的范围是控制/通信逻辑及标称仿真；不代表真机末帧平衡或任意时刻切换验证。
