
%% ׼�������ռ�
clear ; close all; clc
global outlayer2;
global inlayer3;
global inlayer2;
global outlayer3;
global outlayer1;
%% ��������
HandWrittenNumberDatasePath = fullfile('./','/HandWrittenNumberDataset/');
imds = imageDatastore(HandWrittenNumberDatasePath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
labels=imds.Labels;%�����ǩ����
%% ���ݼ�ͼ����
% ����matlab�Դ����������ݼ���Ϊѵ��������֤�������֣��������������
%�����Ҷ�ImageDatastore�������Ͳ���Ϥ����ת��Ϊ��Ϊ��Ϥ�Ľṹ������
%ѵ��������֤���ֱ�Ϊtrains��tests
for i=1:220
    %��ͼ��������ϳ�������
    tempX=im2double(readimage(imds,i));
    trains(i).X=double((tempX(:))');
    %������ͼƬ
    trains(i).X(trains(i).X<0.7)=0;
    trains(i).X(trains(i).X>=0.7)=1;
    tep=double(char(labels(i)))-48;
    if tep==0
        tep=10;
    end
    trains(i).label=tep;
end
%% ����BP������Ľṹ�����������������������2-3�����磩
img = readimage(imds,1);
[sz1,sz2]=size(img);%��ȡͼƬ��С
L=sz1*sz2;%�����ڵ������Ҳ��ͼƬת���������ĳ���
%����Ȩ��ϵ��������ʼ���������
%���ó��������磬�����м��ڵ�Ϊ25
Theta1=(rand(25,L+1)-0.5);
Theta2=(rand(10,26)-0.5);%
%% ѵ�������磨�����������
%����ѧϰ����
n=0.1;
%���ɱ�׼����������
Y=eye(10);
%ѵ��������
i=1;k=1;
maxC=inf;%����������ֵ����ʼ��Ϊ����
error=0.1;%��������������������ԽС��Ԥ������Խ�ã���������Դ��Խ��
while maxC > error
    %����������ֵ
    pred=predict(Theta1,Theta2,trains(i).X);
    C(i)=(outlayer3-Y(:,trains(i).label))'*(outlayer3-Y(:,trains(i).label))/2;
    [~,preddata]=max(pred);
    %���򴫲�������Ȩֵ
    while C(i)>error
        delta3=(outlayer3-Y(:,trains(i).label)).*dsigmoid(inlayer3);%���������delta
        delta2=(delta3'*Theta2(:,2:end))'.*dsigmoid(inlayer2);%����ڶ���delta
        Delta2=delta3*outlayer2';%����Ȩ�ئ�2���ݶ�
        Delta1=delta2*outlayer1;%����Ȩ�ئ�1���ݶ�
        %����Ȩ��
        Theta2=Theta2-n.*Delta2.*C(i);
        Theta1=Theta1-n.*Delta1.*C(i);
        %�ٴμ������ֵ
        pred=predict(Theta1,Theta2,trains(i).X);
        C(i)=(outlayer3-Y(:,trains(i).label))'*(outlayer3-Y(:,trains(i).label))/2;
        [~,preddata]=max(pred);
        K(k)=C(i);
        k=k+1;
    end
    %�����������������ֵ�Ƿ��������
    for t=1:220
        test=predict(Theta1,Theta2,trains(t).X);
        alltest(t)=(outlayer3-Y(:,trains(t).label))'*(outlayer3-Y(:,trains(t).label))/2;
    end
    %�õ���������������ֵ����������������һ�ε���
    [maxC,index]=max(alltest);
    i=index;  
end
%�滭�����򴫲��������ֵ�仯����
    figure(1);
    plot(K);
    hold on;
    title('���򴫲��������ֵ')
    hold off;
%���������ľ���
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
%���Ƴ����򴫲���������ݵ�Ԥ�����
figure(2);
plot(test);
hold on;
accur=accur/220;
title(['��ȷ�ȣ�',num2str(accur)]);
hold off;

%% ����ѵ�����
%�������ѵ������
HandWrittenNumberLabel = fullfile('./','/HandWrittenNumberLabel/');
numtest = imageDatastore(HandWrittenNumberLabel,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
for i=1:10
    %��ͼ��������ϳ�������
    tempX=im2double(readimage(numtest,i));
    tests(i).X=double((tempX(:))');
    tests(i).X(tests(i).X<0.7)=0;
    tests(i).X(tests(i).X>=0.7)=1;
end
%��ʼ���ԣ�����Ԥ����
for i=1:10
pred=predict(Theta1,Theta2,tests(i).X);
[~,preddata]=max(pred);
%����Ԥ��������10����Ϊ0
    if  preddata==10
        preddata=0;
    end
    testlabel(i)=preddata;%testlabel�����¼Ԥ���ֵ
end
%% ��ʾԤ������ʵ��ֵ
n_Sample=10;
figure(3);
for i = 1:n_Sample
    subplot(2,fix((n_Sample+1)/2),i)
    imshow(char(numtest.Files(i)))
    title(['Ԥ��ֵ��' num2str(testlabel(i))])
    xlabel(['��ʵֵ:' num2str(i-1)],'Color','b')
end

%% 
%����Ϊʹ�� HandWrittenNumberDataset�ļ��Ĳ���ѵ��
%% ����ѵ�����
% %�������ѵ������
% HandWrittenNumberDataset = fullfile('./','/HandWrittenNumberDataset/');
% moredatatest = imageDatastore(HandWrittenNumberDataset,...
%     'IncludeSubfolders',true,'LabelSource','foldernames');
% for i=1:220
%     %��ͼ��������ϳ�������
%     tempX=im2double(readimage(moredatatest,i));
%     moretests(i).X=double((tempX(:))');
%     moretests(i).X(moretests(i).X<0.7)=0;
%     moretests(i).X(moretests(i).X>=0.7)=1;
% end
% %��ʼ���ԣ�����Ԥ����
% randr=randperm(220,10);
% for i=1:10
% pred=predict(Theta1,Theta2,moretests(randr(i)).X);
% [~,preddata]=max(pred);
% %����Ԥ��������10����Ϊ0
%     if  preddata==10
%         preddata=0;
%     end
%     testlabel(i)=preddata;%testlabel�����¼Ԥ���ֵ
% end
% %% ��ʾԤ������ʵ��ֵ
% n_Sample=10;
% figure(4);
% for i = 1:n_Sample
%     subplot(2,fix((n_Sample+1)/2),i)
%     imshow(char(moredatatest.Files(randr(i))))
%     title(['Ԥ��ֵ��' num2str(testlabel(i))])
% end