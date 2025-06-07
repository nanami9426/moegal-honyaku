# 快速开始

## 浏览器插件安装

1. 首先下载Chrome浏览器或Edge浏览器。
2. 下载插件代码：

~~~bash
git clone https://github.com/nanami9426/moegal-honyaku-fe.git
~~~

1. 将克隆的插件代码添加到浏览器的自定义插件中。

## 后端代码启动

1. 克隆后端代码
   ~~~bash
   git clone https://github.com/nanami9426/moegal-honyaku
   ~~~

2. 安装环境管理工具
   ~~~bash
   # On macOS and Linux.
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows.
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # With pip.
   pip install uv
   ~~~

3. 安装项目依赖
   ~~~bash
   gmake install
   ~~~

4. 配置环境变量

   ~~~bash
   mv .env.template .env
   # 然后将.env文件中的相关值修改成自己想要的就可以了
   ~~~

5. 运行项目
   ~~~bash
   gmake run
   ~~~

   

