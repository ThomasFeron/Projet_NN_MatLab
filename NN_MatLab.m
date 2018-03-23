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
%    O   O   O   O      O ?
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

learning_rate = 2;

L1=4;  %number of neurons in first layer
L2=4;  %number of neurons in second layer
L3=10; %number of Output neurons Limited to sie of label = 2

%initial weights matrix (randomized normal distribution centered on 0)
w1=randn(L1,length(Input.data));  % L1 x image vector length
w2=randn(L2,L1);
w3=randn(L3,L2);

% NN visualisation
if plotflag==1
fig=figure('color',[0.298 0.6 0],'position',[100,70,1100,500]);
end

%% %%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%

for image_num = 1:1115
   
    Input = FetchInput(image_num,train);
    
    % First Layer
    for i =1:L1
    N1(i)= sigmoid(sum(Input.data.*w1(i,:)),0);
    end

    % Second Layer
    for i =1:L2
    N2(i)= sigmoid(sum(N1.*w2(i,:)),0);
    end

    % Last Layer (output)
    for i =1:L3
    Output(i) = sigmoid(sum(N2.*w3(i,:)),0);
    end
    
    if plotflag ==1
    % plot visualisation
    View_NN(Input,N1,N2,Output)
        if j==1 && image_num ==1
            pause
        else
            drawnow
        end
    end
    
    % Effectivness    
    Cost=(Input.label-Output)';

%     Cost=1/length(Input.label).*sum((Input.label-Output)'.^2);
    
%     Delta=-(Input.label-Output)*Output*(1-Output)*N3;
    %% Back propagation
    
    w3_step=Cost.*sigmoid(Output,1)*learning_rate;
    for k=1:L3
    w3(k,:)=w3(k,:)+(N2*w3_step(k));
    end
    
    w2_step=Cost.*sigmoid(N2,1)*learning_rate;
    for k=1:L2
    w2(k,:)=w2(k,:)+(N1*w2_step(k));
    end

    w1_step=Cost.*sigmoid(N1,1)*learning_rate;
    for k=1:L1
    w1(k,:)=w1(k,:)+(Input.data*w1_step(k));
    end
    

end
disp('Done Training')

%% %%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%%

train=0;
Correct=0;
Miss=0;

for image_num = 1:477
   
    Input = FetchInput(image_num,train);
    
    % First Layer
    for i =1:L1
    N1(i)= sigmoid(sum(Input.data.*w1(i,:)),0);
    end

    % Second Layer
    for i =1:L2
    N2(i)= sigmoid(sum(N1.*w2(i,:)),0);
    end

    % Last Layer (output)
    for i =1:L3
    Output(i) = sigmoid(sum(N2.*w3(i,:)),0);
    end
    
    
    % Score
    if round(Output)==Input.label
        Correct=Correct+1;
    else
        Miss = Miss+1;
    end
    
end
disp('Done Testing')
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
%     length(Input.data),
    m=max([length(N1),length(N2),length(Input.label)]);
    
%     [Input.data'],...

    z = [[zeros(floor((m-length(N1'))/2),1);N1';zeros(round((m-length(N1'))/2),1)],...
        [zeros(floor((m-length(N2'))/2),1);N2';zeros(round((m-length(N2'))/2),1)],...
        [zeros(floor((m-length(Output'))/2),1);Output';zeros(round((m-length(Output'))/2),1)],...
        [zeros(floor((m-length(Input.label'))/2),1);Input.label';zeros(round((m-length(Input.label'))/2),1)]...
        ];

    subplot(1,2,2);
    heatmap(z);
    title('apuyer sur n''importe quelle touche pour commencer')
        
end


