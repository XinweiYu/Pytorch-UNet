function cline_from_python(dataFolder)

%% load eigenbasis
eigenWormFile='/tigress/LEIFER/communalCode/3dbrain/eigenWorms.mat';
load(eigenWormFile);

clines = load([dataFolder filesep 'BehaviorAnalysis/centerline.mat']);

%num_frame = length(clines.centerline);
centerline = clines.centerline;
%centerline = zeros(100,2,num_frame);
% for i=1:num_frame
%     if ~isempty(clines.centerline{i})
%         centerline(:,:,i) = clines.centerline{i}(1:100,:);
%     end
% end
centerline = permute(centerline,[2,3,1]);
%create wormcentered coordinate system
wormcentered=FindWormCentered(centerline);
%project onto eigen basis
eigbasis = imresize(eigbasis,[size(eigbasis,1),99]);
eigenProj=eigbasis*wormcentered;

behaviorFolder=[dataFolder filesep 'BehaviorAnalysis'];
save([behaviorFolder filesep 'centerline'] ,'centerline','eigenProj'...
    ,'wormcentered');
