# 2025Seminar
基于compressAI的端到端图像代码复现总结
## 一、使用说明
1. 训练模型的命令 λ=[0.0018, 0.0054, 0.0162, 0.0483]可以自行选择
```bash
python train.py --model [模型名] --lambda [lambda值]
```
2. 测试模型的命令
```bash
python test.py --model_name [模型名] --path [模型路径]
```
3. 绘图的命令，其中-m默认为计算ms-ssim，具体绘图参数可在代码中修改。关于RD与RD3的不同，详见“二、代码组成”的第5项。
```bash
python /code/code/drawRD.py --input-dir /code/reconstruction -m psnr --show
python /code/code/drawRD3.py -m psnr --show
```
4. 训练结果：JSON文件包含在reconstruction和reconstruction2文件夹中，绘图和BD-RATE计算结果保存在results文件夹中。
## 二、代码组成
1. data_utils.py:与数据处理有关的函数工具，包括点云处理，图片裁剪，rgb和yuv转换的函数。
2. data_loader.py:创建dataloader，在复现中使用的是：KodakDataset, CLIC2020, H5Dataset_Flicker2W
在一个dataloader里，需要有初始化init，获取图片getitem，统计图片数量len。
3. train.py:训练代码，以下列出主函数以及一些比较重要的函数与类
### main函数的简化版本
```bash
args = parse_args(argv)                   #获取参数，并验证参数有效性；修改参数和数据集路径都可以在parse_args中修改default值
test/train_dataset = H5Dataset_Flicker2W(...)   #构建数据集和dataloader
test/train_dataloader = DataLoader(...)
#进行初始化，比如device，lr_scheduler，net
#接下来是主函数的核心部分，功能是：从起始epoch一直训练到最后epoch，最后根据train和test的结果进行best_loss的更新，保留checkpoint
for epoch in range(last_epoch, args.epochs):
```
### 计算与优化指标
```bash
def mse2psnr
class RateDistortionLoss
def configure_optimizers #构建optimizer, aux_optimizer两个优化器，调整学习率
```
### 训练函数
```bash
def train_one_epoch
#基本流程
#开启model.train()之后，初始化一系列参数，loss、bpp_loss等等，接下来开始从dataloader中获取数据
for i, d in enumerate(train_dataloader):
      global_step+=1                    #这是train_one_epoch返回的迭代次数
      d = d.to(device)                  #将数据上传GPU
      optimizer.zero_grad()
      aux_optimizer.zero_grad()         #梯度置零
      out_net = model(d)                #设置模型网络，前向
      out_criterion = criterion(out_net, d)     #计算训练得到的结果
      out_criterion["loss"].backward()          #梯度反传
      
      if clip_max_norm > 0:              #这个操作是防止梯度爆炸
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            if total_norm.isnan() or total_norm.isinf():
                print("non-finite norm, skip this batch")
                continue
      optimizer.step()                    #更新optimizer参数
                                          #aux_optimizer和optimizer的过程基本一致
      aux_loss = model.aux_loss()
      aux_loss.backward()
      aux_optimizer.step()
                                          #更新bpp_loss等一系列参数
      bpp_loss.update(out_criterion["bpp_loss"])
      #......
      if i % 100 == 0 :
          #打印数据
```
### 验证函数
```bash
def test_epoch
#这个函数和train_one_epoch相似，模型需要换成eval状态
model.eval()
device = next(model.parameters()).device #确保使用相同的设备
#初始化参数
#开始验证，这里的代码和train_one_epoch的流程基本一致，不赘述
with torch.no_grad(): #禁用梯度
    for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            bpp_loss.update(out_criterion["bpp_loss"])
            #......
#最后进行打印和数据保存
```
### 其他函数
```bash
def save_checkpoint #保存模型检查点，这样就不需要从头训练了
def parse_args      #获取参数与数据路径，可自行修改
```
4. test.py:测试模型性能的函数。这个函数对测试集的图片进行处理，然后输入到模型中进行测试、计算指标(psnr/ms-ssim/bpp)，最后将结果写成JSON文件(复现时添加了保存json文件的功能，便于绘图时使用)。
 ```bash
   #输出文件路径，文件名的0.0代表lambda
    output_filename = f"/output/{args.model_name}_results_0.0.json"
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)
 ```
5. drawRD.py与drawRD3.py:这两个都是绘制RD曲线的函数，最初的版本存放在/code/src/utils/plot，不同的是：drawRD.py适用于/code/reconstruction路径下的JSON文件格式，drawRD3.py适用于/code/reconstruction2路径下的JSON文件格式。这两个路径中各包含JSON可供参考。
6. 关于熵模型和补全代码的分析与理解已写在代码注释中，此处不赘述。
