
%% 准备工作空间
clear ; close all; clc
global outlayer2;
global inlayer3;
global inlayer2;
global outlayer3;
global outlayer1;
%% 导入数据
HandWrittenNumberDatasePath = fullfile('./','/HandWrittenNumberDataset/');
imds = imageDatastore(HandWrittenNumberDatasePath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
labels=imds.Labels;%导入标签数据
%% 数据集图个数
% 利用matlab自带函数把数据集分为训练集和验证集两部分：（补充代码区）
%由于我对ImageDatastore数据类型不熟悉，故转化为较为熟悉的结构体数据
%训练集和验证集分别为trains和tests
for i=1:220
    %将图像矩阵整合成列向量
    tempX=im2double(readimage(imds,i));
    trains(i).X=double((tempX(:))');
    %两极化图片
    trains(i).X(trains(i).X<0.7)=0;
    trains(i).X(trains(i).X>=0.7)=1;
    tep=double(char(labels(i)))-48;
    if tep==0
        tep=10;
    end
    trains(i).label=tep;
end
%% 定义BP神经网络的结构（补充代码区，可自由设置2-3层网络）
img = readimage(imds,1);
[sz1,sz2]=size(img);%读取图片大小
L=sz1*sz2;%输入层节点个数，也是图片转化成向量的长度
%设置权重系数，并初始化成随机数
%设置成三层网络，其中中间层节点为25
Theta1=(rand(25,L+1)-0.5);
Theta2=(rand(10,26)-0.5);%
%% 训练神经网络（补充代码区）
%定义学习速率
n=0.1;
%生成标准结果输出矩阵
Y=eye(10);
%训练神经网络
i=1;k=1;
maxC=inf;%定义最大误差值，初始化为无穷
error=0.1;%设置最大运行误差，运行误差越小，预测结果就越好，但所需资源就越大
while maxC > error
    %正向计算误差值
    pred=predict(Theta1,Theta2,trains(i).X);
    C(i)=(outlayer3-Y(:,trains(i).label))'*(outlayer3-Y(:,trains(i).label))/2;
    [~,preddata]=max(pred);
    %反向传播，调整权值
    while C(i)>error
        delta3=(outlayer3-Y(:,trains(i).label)).*dsigmoid(inlayer3);%计算第三层delta
        delta2=(delta3'*Theta2(:,2:end))'.*dsigmoid(inlayer2);%计算第二层delta
        Delta2=delta3*outlayer2';%计算权重θ2的梯度
        Delta1=delta2*outlayer1;%计算权重θ1的梯度
        %更新权重
        Theta2=Theta2-n.*Delta2.*C(i);
        Theta1=Theta1-n.*Delta1.*C(i);
        %再次计算误差值
        pred=predict(Theta1,Theta2,trains(i).X);
        C(i)=(outlayer3-Y(:,trains(i).label))'*(outlayer3-Y(:,trains(i).label))/2;
        [~,preddata]=max(pred);
        K(k)=C(i);
        k=k+1;
    end
    %测试所有样本的误差值是否符合条件
    for t=1:220
        test=predict(Theta1,Theta2,trains(t).X);
        alltest(t)=(outlayer3-Y(:,trains(t).label))'*(outlayer3-Y(:,trains(t).label))/2;
    end
    %得到所有样本误差最大值，和其索引用于下一次调整
    [maxC,index]=max(alltest);
    i=index;  
end
%绘画出反向传播过程误差值变化过程
    figure(1);
    plot(K);
    hold on;
    title('反向传播过程误差值')
    hold off;
%计算调整后的精度
accur=0;
j=1;
for i=1:220
pred=predict(Theta1,Theta2,trains(i).X);
[~,preddata]=max(pred);
    if  preddata==10
        preddata=0;
    end
    test(i)=preddata;
    if test(i)==trains(i).label
        accur=accur+1;
    else
        wrong(j)=i;
        j=j+1;
    end
end
%绘制出反向传播后测试数据的预测情况
figure(2);
plot(test);
hold on;
accur=accur/220;
title(['精确度：',num2str(accur)]);
hold off;

%% 测试训练结果
%导入测试训练数据
HandWrittenNumberLabel = fullfile('./','/HandWrittenNumberLabel/');
numtest = imageDatastore(HandWrittenNumberLabel,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
for i=1:10
    %将图像矩阵整合成列向量
    tempX=im2double(readimage(numtest,i));
    tests(i).X=double((tempX(:))');
    tests(i).X(tests(i).X<0.7)=0;
    tests(i).X(tests(i).X>=0.7)=1;
end
%开始测试，计算预测结果
for i=1:10
pred=predict(Theta1,Theta2,tests(i).X);
[~,preddata]=max(pred);
%调整预测结果，将10更改为0
    if  preddata==10
        preddata=0;
    end
    testlabel(i)=preddata;%testlabel数组记录预测的值
end
%% 显示预测结果与实际值
n_Sample=10;
figure(3);
for i = 1:n_Sample
    subplot(2,fix((n_Sample+1)/2),i)
    imshow(char(numtest.Files(i)))
    title(['预测值：' num2str(testlabel(i))])
    xlabel(['真实值:' num2str(i-1)],'Color','b')
end

%% 
%以下为使用 HandWrittenNumberDataset文件的测试训练
%% 测试训练结果
% %导入测试训练数据
% HandWrittenNumberDataset = fullfile('./','/HandWrittenNumberDataset/');
% moredatatest = imageDatastore(HandWrittenNumberDataset,...
%     'IncludeSubfolders',true,'LabelSource','foldernames');
% for i=1:220
%     %将图像矩阵整合成列向量
%     tempX=im2double(readimage(moredatatest,i));
%     moretests(i).X=double((tempX(:))');
%     moretests(i).X(moretests(i).X<0.7)=0;
%     moretests(i).X(moretests(i).X>=0.7)=1;
% end
% %开始测试，计算预测结果
% randr=randperm(220,10);
% for i=1:10
% pred=predict(Theta1,Theta2,moretests(randr(i)).X);
% [~,preddata]=max(pred);
% %调整预测结果，将10更改为0
%     if  preddata==10
%         preddata=0;
%     end
%     testlabel(i)=preddata;%testlabel数组记录预测的值
% end
% %% 显示预测结果与实际值
% n_Sample=10;
% figure(4);
% for i = 1:n_Sample
%     subplot(2,fix((n_Sample+1)/2),i)
%     imshow(char(moredatatest.Files(randr(i))))
%     title(['预测值：' num2str(testlabel(i))])
% end