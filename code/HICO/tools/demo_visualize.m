
im_root   = '../images/';
bbox_file = '../anno_bbox.mat';

ld = load(bbox_file);
bbox_train = ld.bbox_train;
bbox_test = ld.bbox_test;
list_action = ld.list_action;

% change this
i = 100;  % image index
j = 1;    % hoi index

% read image
im_file = [im_root 'train2015/' bbox_train(i).filename];
im = imread(im_file);

% display image
figure(1);
imshow(im); hold on;

% display hoi
hoi_id = bbox_train(i).hoi(j).id;
aname = [list_action(hoi_id).vname_ing ' ' list_action(hoi_id).nname];
aname = strrep(aname,'_',' ');
title(aname);

% display bbox
if bbox_train(i).hoi(j).invis
    fprintf('hoi not visible\n');
else
    bboxhuman  = bbox_train(i).hoi(j).bboxhuman;
    bboxobject = bbox_train(i).hoi(j).bboxobject;
    connection = bbox_train(i).hoi(j).connection;
    visualize_box_conn_one(bboxhuman, bboxobject, connection, 'b','g');
end

