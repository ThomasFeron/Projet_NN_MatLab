clear
%% Neural Network (Option 10)
% 
%  Input
%    |   N1  N2  Output
%    | w1| w2| w3|
%    O   |   |   |
%    O \ |   |   |
%    O   O - O   |     Labels
%    O   O   O \ |      |
%    O   O   O   O      O
%    O   O   O   O      O
%    O   O   O /
%    O   O - O 
%    O /
%    O

plotflag=0;

%% Input
image_num = 1;
train = 1;
Input = FetchInput(image_num,train); % Feches input image vector from the training lot

%% Construction of the NN

learning_rate = 1;

L1=5;  %number of neurons in first layer
L2=4;  %number of neurons in second layer
L3=10; %number of Output neurons Limited to sie of label = 2

%initial weights matrix (randomized normal distribution centered on 0)
w1=randn(length(Input.data),L1);  % L1 x image vector length
w2=randn(L1,L2);
w3=randn(L2,L3);

% NN visualisation
if plotflag==1
fig=figure('color',[0.298 0.6 0],'position',[100,70,1100,500]);
end

%% %%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%

for image_num = 1:1115 % = 70% 
   
    Input = FetchInput(image_num,train);
    
    % First Layer
    for i =1:L1
    N1(i,1)= sigmoid(sum(Input.data'.*w1(:,i)),0);
    end

    % Second Layer
    for i =1:L2
    N2(i,1)= sigmoid(sum(N1.*w2(:,i)),0);
    end

    % Last Layer (output)
    for i =1:L3
    Output(i,1) = sigmoid(sum(N2.*w3(:,i)),0);
    end
    
    if plotflag ==1 % plot visualisation
    View_NN(Input,N1,N2,Output)
        if image_num ==1
            pause
        else
            drawnow
        end
    end
    
    % Effectivness
    Cost = Output - Input.label';
    Total_Error(image_num) = sum((Input.label-Output).^2);
    
    %% Back propagation   Not working :D
    
    % For w3
    lamda=Output.*(1-Output).*-(Input.label-Output);
    delta_w3=N2*lamda';
    w3 = w3 - delta_w3 * learning_rate;
    
    % For w2
    lamda2= N2.*(1-N2).*(w3*lamda);
    delta_w2 = N1*lamda2';
    w2 = w2 - delta_w2 * learning_rate;
    
    % For w1
    lamda3=N1.*(1-N1).*(w2*lamda2);
    delta_w1 = Input.data'*lamda3';
    w1 = w1 - delta_w1 * learning_rate;
    

end
figure
plot(1:1115,Total_Error)
title('Total Error evolution while training')
%% %%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%%

train=0;
Correct=0;
Miss=0;

for image_num = 1:477 % length of training set = untouched 30% remaining from the database
   
    Input = FetchInput(image_num,train);
    
    % First Layer
    for i =1:L1
    N1(i,1)= sigmoid(sum(Input.data'.*w1(:,i)),0);
    end

    % Second Layer
    for i =1:L2
    N2(i,1)= sigmoid(sum(N1.*w2(:,i)),0);
    end

    % Last Layer (output)
    for i =1:L3
    Output(i,1) = sigmoid(sum(N2.*w3(:,i)),0);
    end
    
    
    % Score
    if round(Output)==Input.label
        Correct=Correct+1;
    else
        Miss = Miss+1;
    end
    
end
disp('Test results: ')
fprintf('Correct %i \n',Correct)
fprintf('Miss %i \n', Miss)

%% %%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%

function y=sigmoid(x,deriv)
% non linear transformation for neurons activations
    if(deriv==1)
        y = x.*(1-x);
    else
        y = 1./(1+exp(-x));
    end
end

function View_NN(Input,N1,N2,Output)
% Displays the input image and the NN neurons activations.

    dim = sqrt(length(Input.data)); % dimention of the image = sqrt of the image vector.
    image = reshape(Input.data,dim,dim);
    
    subplot(1,2,1);
    imshow(image');
    m=max([length(N1),length(N2),length(Input.label)]);

    z = [[zeros(floor((m-length(N1'))/2),1);N1';zeros(round((m-length(N1'))/2),1)],...
        [zeros(floor((m-length(N2'))/2),1);N2';zeros(round((m-length(N2'))/2),1)],...
        [zeros(floor((m-length(Output'))/2),1);Output';zeros(round((m-length(Output'))/2),1)],...
        [zeros(floor((m-length(Input.label'))/2),1);Input.label';zeros(round((m-length(Input.label'))/2),1)]...
        ];

    subplot(1,2,2);
    heatmap(z);
    title('apuyer sur n''importe quelle touche pour commencer')
        
end
