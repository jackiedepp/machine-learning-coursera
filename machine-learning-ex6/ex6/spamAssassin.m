%% a span classifier of spamAssassin

%% ================= Part 1: Get data =======================
X1 = [];
y1 = [];
dir_name = 'spamAssassin/exam';
listing = dir(dir_name);
for i = 3: length(listing)
  filename      = fullfile(dir_name, listing(i).name);
  file_contents = readFile(filename);
  word_indices  = processEmail(file_contents);
  features      = emailFeatures(word_indices);
  X1            = [X1; features'];
end
y1 = [y1; ones(size(X1,1),1)];

X2 = [];
y2 = [];
dir_name = 'spamAssassin/exam_1';
listing = dir(dir_name);
for i = 3: length(listing)
  filename      = fullfile(dir_name, listing(i).name);
  file_contents = readFile(filename);
  word_indices  = processEmail(file_contents);
  features      = emailFeatures(word_indices);
  X2            = [X2; features'];
end
y2 = [y2; zeros(size(X2,1),1)];

% get the whole data
X = [X1; X2];
y = [y1; y2];
%size(X)
%size(y)

% divide up the dataset into a training set, a validation set and a test set
m = size(X, 1);
train_num = floor(m * 0.6);
valid_num = floor(m * 0.3);

rand_indices = randperm(m);
Xtrain = X(rand_indices(1:train_num), :);
ytrain = y(rand_indices(1:train_num), :);
Xvalid = X(rand_indices((train_num+1):(train_num+valid_num)), :);
yvalid = y(rand_indices((train_num+1):(train_num+valid_num)), :);
Xtest  = X(rand_indices((train_num+valid_num+1):end), :);
ytest  = y(rand_indices((train_num+valid_num+1):end), :);


%% ============== Part 2: Training data =========================
C = 0.1; sigma = 0.1;
model = svmTrain(Xtrain, ytrain, C, @(x1,x2) gaussianKernel(x1,x2,sigma));

[C, sigma] = dataset3Params(Xtrain, ytrain, Xvalid, yvalid);
model = svmTrain(Xtrain, ytrain, C, @(x1,x2) gaussianKernel(x1,x2,sigma));

%% ============== Part 3: Predict data ==========================
ptrain = svmPredict(model, Xtrain);
pvalid = svmPredict(model, Xvalid);
ptest  = svmPredict(model, Xtest);
fprintf('Train Accuracy: %f\n', mean(double(ptrain == ytrain)) * 100);
fprintf('Valid Accuracy: %f\n', mean(double(pvalid == yvalid)) * 100);
fprintf('Test  Accuracy: %f\n', mean(double(ptest  == ytest))  * 100);
