function p = predict(Theta1, Theta2, X)
global outlayer1;
global outlayer2;
global outlayer3;
global inlayer2;
global inlayer3;
% 在给定训练的神经网络的情况下预测输入的标签
% p = PREDICT(Theta1, Theta2, X) 在给定训练的神经网络权重（Theta1，Theta2）的情况下输出X的预测标签

m = size(X, 1);
num_labels = size(Theta2, 1);
Y=eye(10); 
%Y=[Y(:,10),Y];
% 补充代码区。
% 将p设置为包含1到num_labels之间标签的向量。
% 提示: 建议使用max函数. max函数也可以返回max元素的索引，有关更多信息，请参阅“help max”。
for i=1:m
    outlayer1=[1,X(i,:)];%为第一层的输入值，大小为1*m+1的行向量
    inlayer2=Theta1*outlayer1';%计算中间层，ininininlayer2为一个25*1的列向量
    outlayer2=[1;sigmoid(inlayer2)];%计算输出到第三层的数值,为一个26*1的列向量
    inlayer3=Theta2*outlayer2;%计算输出层，inlayer3为一个10*1的列向量
    outlayer3=sigmoid(inlayer3);
    [~,a(i)]=max(outlayer3);
    p(:,i)=Y(:,a(i));
end
%返回一个10*1的列向量
end
