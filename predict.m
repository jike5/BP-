function p = predict(Theta1, Theta2, X)
global outlayer1;
global outlayer2;
global outlayer3;
global inlayer2;
global inlayer3;
% �ڸ���ѵ����������������Ԥ������ı�ǩ
% p = PREDICT(Theta1, Theta2, X) �ڸ���ѵ����������Ȩ�أ�Theta1��Theta2������������X��Ԥ���ǩ

m = size(X, 1);
num_labels = size(Theta2, 1);
Y=eye(10); 
%Y=[Y(:,10),Y];
% �����������
% ��p����Ϊ����1��num_labels֮���ǩ��������
% ��ʾ: ����ʹ��max����. max����Ҳ���Է���maxԪ�ص��������йظ�����Ϣ������ġ�help max����
for i=1:m
    outlayer1=[1,X(i,:)];%Ϊ��һ�������ֵ����СΪ1*m+1��������
    inlayer2=Theta1*outlayer1';%�����м�㣬ininininlayer2Ϊһ��25*1��������
    outlayer2=[1;sigmoid(inlayer2)];%������������������ֵ,Ϊһ��26*1��������
    inlayer3=Theta2*outlayer2;%��������㣬inlayer3Ϊһ��10*1��������
    outlayer3=sigmoid(inlayer3);
    [~,a(i)]=max(outlayer3);
    p(:,i)=Y(:,a(i));
end
%����һ��10*1��������
end
