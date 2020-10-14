function [  ] = visualize_box_conn_one( box_1, box_2, conn, clr_1, clr_2 )

if nargin < 4
    clr_1 = 'b';
end

if nargin < 5
    clr_2 = 'g';
end

for r = 1:numel(box_1)
    rt = [box_1(r).x1, ...
        box_1(r).y1, ...
        box_1(r).x2-box_1(r).x1+1, ...
        box_1(r).y2-box_1(r).y1+1];
    rectangle('Position', rt, 'LineWidth', 3, 'EdgeColor', clr_1);
end
for r = 1:numel(box_2)
    rt = [box_2(r).x1, ...
        box_2(r).y1, ...
        box_2(r).x2-box_2(r).x1+1, ...
        box_2(r).y2-box_2(r).y1+1];
    rectangle('Position', rt, 'LineWidth', 3, 'EdgeColor', clr_2);
end
for c = 1:size(conn,1)
    rt1 = box_1(conn(c,1));
    rt2 = box_2(conn(c,2));
    ct1 = [(rt1.x1+rt1.x2)/2, (rt1.y1+rt1.y2)/2];
    ct2 = [(rt2.x1+rt2.x2)/2, (rt2.y1+rt2.y2)/2];
    
    line( ...
        'XData', [ct1(1); ct2(1)], ...
        'YData', [ct1(2); ct2(2)], ...
        'Color', 'r', ...
        'LineStyle', '-', ...
        'LineWidth', 3 ...
        );
    plot(gca, ct1(1), ct1(2), 'ro', 'LineWidth', 2, 'MarkerFaceColor', 'red');
    plot(gca, ct2(1), ct2(2), 'ro', 'LineWidth', 2, 'MarkerFaceColor', 'red');
end

end

