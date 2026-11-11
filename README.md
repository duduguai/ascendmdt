<font face="Times New Roman">

<div align="center">

# Ascend-Model-Deploy-ToolKit

</div>

---

## 简介

`Ascend-Model-Deploy-ToolKit`（简称 `AscendMDT`）是我们面向昇腾 NPU（当前主要覆盖 `310B / P` 型号）开发的模型部署工具库。该工具库采用三级分层架构：一级聚焦 NPU 资源分配与运行，负责 NPU 运行环境管理、运行资源申请与释放，以及 NPU 调度与执行；二级聚焦模型功能实现，负责各模型特有的数据处理流程、数据搬运，并调用一级结构提供的资源与运行能力；三级面向外部提供模型封装，将同类模型抽象为统一基类，具体模型通过继承基类实现模型接口的创建与调用。模型推理逻辑封装在工具库内部，对外提供一行式调用接口，以完成模型的部署与推理。

---

## 目录

<details open>
<summary>Clik to expand project directory architecture</summary>

```bash
ascendmdt
├── CMakeLists.txt                          # ascendmdt 编译构造 CMakeLists
├── README.md                               # 说明文档
├── build.sh                                # ascendmdt 编译构造执行脚本
├── inc                                     # ascendmdt 头文件目录
│   ├── aclsession                          # 一级结构 : ACL会话层
│   │   ├── core                            # ACL 会话层内核
│   │   │   ├── aclsession_core.h
│   │   │   └── types_session_core.h
│   │   ├── general                         # ACL 会话层通用派生
│   │   │   ├── aclsession_general.h
│   │   │   └── types_session_general.h
│   │   └── yolo                            # ACL 会话层针对 YOLO 派生
│   │       ├── aclsession_yolo.h
│   │       └── types_session_yolo.h
│   ├── modelcore                           # 二级结构 : 模型功能层
│   │   └── yolo                            # YOLO 模型功能层
│   │       ├── models_core_yolo.h
│   │       └── types_models_yolo.h
│   ├── models                              # 三级结构 : 模型封装层
│   │   └── yolo                            # YOLO 模型封装层
│   │       ├── commontypes_yolo.h
│   │       ├── models_yolo_base.h          # YOLO 模型抽象基类
│   │       └── models_yolo_list.h          # YOLO 模型调用清单
│   └── tools                               # 工具定义
│       ├── tools_base.h                    # 通用工具
│       └── tools_cv.h                      # 适用于 CV 的工具
├── src                                     # ascendmdt 源文件目录
│   ├── aclsession                          # 一级结构 : ACL 会话层实现
│   │   ├── aclsession_core.cpp
│   │   ├── aclsession_general.cpp
│   │   └── aclsession_yolo.cpp
│   ├── modelcore                           # 二级结构 : 模型功能层实现
│   │   └── models_core_yolo.cpp
│   └── models                              # 三级结构 : 模型封装层实现
│       └── models_yolo.cpp
└── tests                                   # 测试目录
    ├── build.sh                            # 测试程序构建执行脚本
    ├── CMakeLists.txt                      # 测试程序构建 CMakeLists
    ├── src                                 # 测试程序源文件
    │   └── test.cpp
    └── test.sh                             # 测试程序执行脚本
```

</details>

---

## 构建

`ascendmdt` 工具库的构建可通过本仓库提供的构建脚本完成。在本仓库的根目录下执行 `build.sh` 脚本，会自动将 `ascendmdt` 的源代码进行编译并生成动态库保存在目录 `ROOT/lib` 下。构建脚本会自动识别当前编译主机的架构，将动态库文件分别对应的保存在 `aarch64-linux` 和 `x86_64-linux` 子目录下。目前 `ascendmdt` 仅完成了 `YOLO` 全系列模型的部署验证，因此当前状态构建仅会生成 `lib_yolo.so` 动态库文件。

---

## 更新说明

- `2026-11-06`: 完成 `ascendmdt` 框架创建, 添加全系列 `YOLO` 模型调用推理入口，新增 `Ultralytics` 的 `YOLOv8、11、26` 启用实例分割功能入口.
- `2026-10-28`: 完成 `ascendmdt` 调用 `YOLO` 系列模型推理部署的功能通路验证. 

---

## 后续开发

- **目录说明**
    - `aclsession` 目录下存放一级结构 ACL会话层的定义，其中 `core` 目录是 ACL 会话层的内核定义，定义了 ACL 会话的通用接口与处理逻辑；`general` 与 `yolo` 目录下均是具体的 ACL 会话实现；后续需要新增 ACL 会话层面的支持，推荐的方式是基于 `core` 完成对应功能 ACL 会话层派生。
    - `modelcore` 目录下存放二级结构模型功能层的定义，其中 `yolo` 目录是 `YOLO` 系列模型功能实现定义；后续新增模型功能层，在 `modelcore` 新起目录完成功能层定义。
    - `models` 目录下存放的是三级结构模型封装层的定义，其中 `yolo` 目录提供了当前能供通过 `ascendmdt` 实现模型调用推理的全部 `YOLO` 模型清单；该目录也是后续 `ascendmdt` 向外提供的调用入口；后续新增模型需要在该目录完成模型调用清单创建。

- **拟定计划**
    - `ascendmdt` 新增常用分类模型 (包含 `ResNet`, `MobileNet`, `ShuffleNet` 等) 推理部署调用入口.
    - `ascendmdt` 新增常见实例分割模型 (包含 `UNet`系列, `Sam`系列等) 推理部署调用入口.