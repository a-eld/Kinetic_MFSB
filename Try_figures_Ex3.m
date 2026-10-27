close all;

resultsFile = fullfile( ...
    pwd, ...
    'saved_results', ...
    'Ex1_Morse_full_results.mat');

if ~isfile(resultsFile)
    error('Results file not found:\n%s',resultsFile);
end

load(resultsFile);

fprintf('Loaded results from:\n%s\n',resultsFile);

%%
rhoFloor = 1E-14;
sigma2 = params.sigma2;
sigma  = sqrt(sigma2);

xMin = params.xMin;
xMax = params.xMax;
vMin = params.vMin;
vMax = params.vMax;
D    = params.D;

Nx = params.Nx;
Nv = params.Nv;
Nt = params.Nt;

dx = params.dx;
dv = params.dv;
dt = params.dt;

C_R   = params.C_R;
C_A   = params.C_A;
ell_R = params.ell_R;
ell_A = params.ell_A;

bcMu   = params.bcMu;
posTol = params.posTol;

lastOuter = diagnostics.lastOuter;
errOutPlot = diagnostics.errOut;
massErrPlot = diagnostics.massErr;

Lx = xMax-xMin;


dtHalf = 0.5*dt;
AvHalfInt = diffusion_v_matrix(Nv,dtHalf,D,dv);
AvHalfIntDec = decomposition(AvHalfInt,'lu');


if ~exist('Kmat','var')
    S = periodic_distance_matrix(x);
    Kmat = morse_kernel_matrix(S,Lx,C_R,C_A,ell_R,ell_A);
end

%% Nominal Morse evolution
muNom = bcMu*ones(Nx,Nv,Nt+1);
muNom(:,:,1) = impose_phase_boundary(mu_in,bcMu);

massNom = zeros(Nt+1,1);
massNom(1) = sum(muNom(:,:,1),'all')*dx*dv;

clipMassUNC = zeros(Nt,1);

for n = 1:Nt

    U0 = muNom(:,:,n);

  
    [~,g0] = morse_force_coefficients(U0,Kmat,dx,dv);

    [Tpred,~] = morse_pullback(x,v,g0,dt);

    Gpred = zeros(Nx,1);

    Upred = strang_interacting( ...
        U0,AvHalfIntDec,Tpred,Gpred,dt,D,dv,bcMu,posTol, ...
        sprintf('Nominal predictor n=%d',n));

    [~,g1] = morse_force_coefficients(Upred,Kmat,dx,dv);
    gHalf = 0.5*(g0+g1);

    [Thalf,clipMask] = morse_pullback(x,v,gHalf,dt);

    Ghalf = zeros(Nx,1);

  
    U1 = strang_interacting( ...
        U0,AvHalfIntDec,Thalf,Ghalf,dt,D,dv,bcMu,posTol, ...
        sprintf('Nominal corrected n=%d',n));

    muNom(:,:,n+1) = U1;

    massNom(n+1) = sum(U1,'all')*dx*dv;
    clipMassUNC(n) = sum(U1.*clipMask,'all')*dx*dv;
end

fprintf('\nNominal Morse evolution:\n');
fprintf('maximum mass error   = %.3e\n',max(abs(massNom-massNom(1))));
fprintf('maximum clipped mass = %.3e\n',max(clipMassUNC));
fprintf('final mass            = %.12f\n',massNom(end));

%%
plotTimes = linspace(0,1,5);
plotIdx = round(plotTimes*Nt) + 1;

xPlotMin = -2; 
xPlotMax =  2;
vPlotMin = -3;
vPlotMax =  3;

ixPlot = find(x >= xPlotMin & x <= xPlotMax);
ivPlot = find(v >= vPlotMin & v <= vPlotMax);

[X0,V0] = meshgrid(x(ixPlot),v(ivPlot));



zMax = 0;
cMin = Inf;
cMax = -Inf;

for pidx = 1:numel(plotIdx)
    n = plotIdx(pidx);

    Z0tmp = mu(ixPlot,ivPlot,n)';
   
    zMax = max(zMax,max(Z0tmp,[],'all'));

    cMin = min(cMin,min(Z0tmp,[],'all'));
    cMax = max(cMax,max(Z0tmp,[],'all'));
end

if cMax <= cMin
    cMax = cMin + eps;
end

zTop = 1.005*zMax;


zTickTop = ceil(10*zTop)/10;
if zTickTop <= 0
    zTickTop = 1;
end

fig = figure('Position',[50 100 2600 500]);

nPanels = numel(plotIdx);

leftMargin   = 0.05;
rightMargin  = 0.05;
bottomMargin = 0.3;
topMargin    = 0.8;

gap = -0.06;   

panelW = (1 - leftMargin - rightMargin - (nPanels-1)*gap)/nPanels;
panelH = topMargin - bottomMargin;

axList = gobjects(nPanels,1);

for pidx = 1:nPanels

    surfacePanelW = 1.4*panelW;

    totalSurfaceWidth = ...
        (nPanels-1)*(panelW+gap) + surfacePanelW;

    surfaceLeftMargin = 0.5*(1-totalSurfaceWidth);

    ax = axes(fig,'Position',[ ...
        surfaceLeftMargin + (pidx-1)*(panelW+gap), ...
        bottomMargin, ...
        surfacePanelW, ...
        1.4*panelH]);

    

    axList(pidx) = ax;

    n = plotIdx(pidx);

    Z0tmp = mu(ixPlot,ivPlot,n)';

    [Xq,Vq] = meshgrid( ...
        linspace(x(ixPlot(1)),x(ixPlot(end)),200), ...
        linspace(v(ivPlot(1)),v(ivPlot(end)),300));

    Z_dense = interp2( ...
        X0,V0,Z0tmp, ...
        Xq,Vq, ...
        'cubic');
    s = surf(ax,Xq,Vq,Z_dense, ...
        'EdgeColor','none', ...
        'FaceColor','interp', ...
        'FaceLighting','gouraud');



    shading(ax,'interp');
    colormap(fig,turbo);

    clim(ax,[cMin,1.7]);

    view(ax,-45,20);
    set(ax,'Projection','perspective');

    lighting(ax,'gouraud');
    s.SpecularStrength = 0.3; 
    s.DiffuseStrength = 0.75;
    s.SpecularExponent = 20;
    camlight(ax,'right');

    axis(ax,'tight');
    xlim(ax,[xPlotMin,xPlotMax]);
    ylim(ax,[vPlotMin,vPlotMax]);
    zlim(ax,[0,zTickTop]);

    pbaspect(ax,[1 1 1.1]);

    grid(ax,'off');
    box(ax,'off');

    

    if pidx == 1

    set(ax, ...
        'FontSize',22, ...
        'TickLabelInterpreter','latex', ...
        'XTick',[-2 0 2], ...
        'YTick',[-3 0 3], ...
        'ZTick',[], ...
        'TickDir','out');

    % Axis labels in their normal positions below the 3D axes
    xlabel(ax,'$x$', ...
    'Interpreter','latex', ...
    'FontSize',27, ...
    'Rotation',0);

ylabel(ax,'$v$', ...
    'Interpreter','latex', ...
    'FontSize',27, ...
    'Rotation',0);

ax.ZColor = 'none';

drawnow;

% Move numerical tick labels slightly closer
ax.XAxis.TickLabelGapOffset = -5;
ax.YAxis.TickLabelGapOffset = -5;

% Move the x and v labels slightly closer to their axes
ax.XLabel.Units = 'normalized';
ax.YLabel.Units = 'normalized';

xLabelPos = ax.XLabel.Position;
yLabelPos = ax.YLabel.Position;

xLabelPos(2) = xLabelPos(2) + 0.04;   % move x upward
yLabelPos(1) = yLabelPos(1) + 0.04;   % move v rightward

ax.XLabel.Position = xLabelPos;
ax.YLabel.Position = yLabelPos;

ax.XLabel.HorizontalAlignment = 'center';
ax.YLabel.HorizontalAlignment = 'center';

else

    set(ax, ...
        'XTick',[], ...
        'YTick',[], ...
        'ZTick',[]);

    ax.XColor = 'none';
    ax.YColor = 'none';
    ax.ZColor = 'none';

end
    % time label on top of each panel
    ttl = title(ax,sprintf('$t = %.2f$',t(n)), ...
        'Interpreter','latex', ...
        'FontSize',25);
    ttl.Units = 'normalized';
    ttl.Position = [0.5 0.9 0];
end

drawnow;
exportgraphics(fig,'Ex3-Morse-phase_space_surfaces.pdf','Resolution',300);


%% Heat maps of mu_t with total phase-space drift streamlines

fig = figure('Position',[50 100 2600 500]);

axList = gobjects(nPanels,1);

for pidx = 1:nPanels

    ax = axes(fig,'Position',[ ...
        leftMargin + (pidx-1)*(panelW + gap) + 0.10*panelW, ...
        bottomMargin + 0.10*panelH, ...
        0.6*panelW, ...
        0.9*panelH]);

    axList(pidx) = ax;

    n = plotIdx(pidx);

    U = mu(:,:,n);

   logphi_INT  = log(max(varphi(:,:,n),rhoFloor));
   dlogphi_INT = centered_Dv_2D(logphi_INT,dv);


    u_INT = sigma2*dlogphi_INT;

    [~,~,FINT] = ...
        morse_force_coefficients(U,Kmat,dx,dv);

    % Heat map of mu_t
    imagesc(ax, ...
        x(ixPlot), ...
        v(ivPlot), ...
        U(ixPlot,ivPlot)');

    axis(ax,'xy');

    colormap(fig,turbo);
    clim(ax,[cMin,1.7]);

    hold(ax,'on');


    hs = streamslice(ax, ...
        X0,V0, ...
        V0, ...
        FINT(ixPlot,ivPlot)' + ...
        u_INT(ixPlot,ivPlot)', ...
        0.5);
     set(hs, ...
        'Color',[1.0 0.5 0.0], ...
        'LineWidth',1);

    hold(ax,'off');

    xlim(ax,[xPlotMin xPlotMax]);
    ylim(ax,[vPlotMin vPlotMax]);

    if pidx == 1
        axis(ax,'on');
        box(ax,'on');

        set(ax, ...
            'FontSize',18, ...
            'TickLabelInterpreter','latex', ...
            'XTick',[-2 0 2], ...
            'YTick',[-3 0 3], ...
            'TickDir','out', ...
            'XColor','k', ...
            'YColor','k');

        text(ax,0.50,1.04,'$x$', ...
            'Units','normalized', ...
            'Interpreter','latex', ...
            'FontSize',27, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', ...
            'Clipping','off');

        ylabel(ax,'$v$', ...
            'Interpreter','latex', ...
            'FontSize',27, ...
            'Rotation',0);

        ax.YLabel.Units = 'normalized';
        ax.YLabel.Position = [-0.18 0.50 0];
        ax.YLabel.HorizontalAlignment = 'center';
        ax.YLabel.VerticalAlignment = 'middle';

    else
        axis(ax,'off');

    end
end


axMid = axList(3);
midPos = axMid.Position;

cb = colorbar(axMid,'southoutside');

cb.FontSize = 22;
cb.TickLabelInterpreter = 'latex';
colorLimits = axMid.CLim;
cb.Ticks = unique([[0,0.4,0.8,1.2,1.6],colorLimits(1)]);
cb.TickLabels = arrayfun( ...
    @(value) sprintf('%.1f',value), ...
    cb.Ticks, ...
    'UniformOutput',false);


cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 27;
cb.Label.Rotation = 0;


axMid.Position = midPos;

drawnow;

leftCB  = axList(2).Position(1);
rightCB = axList(4).Position(1) + axList(4).Position(3);

fullWidth = rightCB-leftCB;

shrinkFactor = 0.65;
cbWidth = shrinkFactor*fullWidth;
cbLeft = leftCB + 0.5*(fullWidth-cbWidth);

cb.Position = [ ...
    cbLeft, ...
    0.19, ...
    cbWidth, ...
    0.035];

drawnow;



pdfFile = 'Ex3-Morse-mu_heatmaps_with_drift_streamslice.pdf';

drawnow;

exportgraphics(fig, ...
    pdfFile, ...
    'ContentType','vector', ...
    'Padding','tight');


%% Nominal density heatmaps with Nominal-drift streamlines

cMin_nom = Inf;
cMax_nom = -Inf;

for pidx = 1:numel(plotIdx)
    n = plotIdx(pidx);

    Z0tmp = muNom(ixPlot,ivPlot,n)';

    cMin_nom = min(cMin_nom,min(Z0tmp,[],'all'));
    cMax_nom = max(cMax_nom,max(Z0tmp,[],'all'));
end

if cMax_nom <= cMin_nom
    cMax_nom = cMin_nom + eps;
end

fig = figure('Position',[50 100 2600 500]);

axList = gobjects(nPanels,1);

for pidx = 1:nPanels

    ax = axes(fig,'Position',[ ...
        leftMargin + (pidx-1)*(panelW + gap) + 0.10*panelW, ...
        bottomMargin + 0.10*panelH, ...
        0.6*panelW, ...
        0.9*panelH]);

    axList(pidx) = ax;

    n = plotIdx(pidx);

    U = muNom(:,:,n);

    [~,~,FUNC] = ...
        morse_force_coefficients(U,Kmat,dx,dv);


    imagesc(ax, ...
        x(ixPlot), ...
        v(ivPlot), ...
        U(ixPlot,ivPlot)');

    axis(ax,'xy');

    colormap(fig,turbo);
    clim(ax,[cMin,1]);

    hold(ax,'on');


    hs = streamslice(ax, ...
        X0,V0, ...
        V0, ...
        FUNC(ixPlot,ivPlot)', ...
        0.5);

    set(hs, ...
        'Color',[1.0 0.5 0.0], ...
        'LineWidth',1);

    hold(ax,'off');

    box(ax,'on');
   if pidx == 1

    axis(ax,'on');

    set(ax, ...
        'FontSize',18, ...
        'TickLabelInterpreter','latex', ...
        'XTick',[-2 0 2], ...
        'YTick',[-3 0 3], ...
        'TickDir','out', ...
        'XColor','k', ...
        'YColor','k');

    text(ax,0.50,1.04,'$x$', ...
        'Units','normalized', ...
        'Interpreter','latex', ...
        'FontSize',27, ...
        'Color','k', ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom', ...
        'Clipping','off');

    ylabel(ax,'$v$', ...
        'Interpreter','latex', ...
        'FontSize',27, ...
        'Color','k', ...
        'Rotation',0);

    ax.YLabel.Units = 'normalized';
    ax.YLabel.Position = [-0.18 0.50 0];
    ax.YLabel.HorizontalAlignment = 'center';
    ax.YLabel.VerticalAlignment = 'middle';

    else

    axis(ax,'off');

   end

end


axMid = axList(3);
midPos = axMid.Position;

cb = colorbar(axMid,'southoutside');

cb.FontSize = 22;
cb.TickLabelInterpreter = 'latex';
colorLimits = axMid.CLim;
cb.Ticks = unique([[0,0.2,0.4,0.6,0.8,1],colorLimits(1),colorLimits(2)]);
cb.TickLabels = arrayfun( ...
    @(value) sprintf('%.1f',value), ...
    cb.Ticks, ...
    'UniformOutput',false);


axMid.Position = midPos;

drawnow;

leftCB  = axList(2).Position(1);
rightCB = axList(4).Position(1) + axList(4).Position(3);

fullWidth = rightCB - leftCB;

shrinkFactor = 0.65;
cbWidth = shrinkFactor*fullWidth;
cbLeft  = leftCB + 0.5*(fullWidth - cbWidth);

cb.Position = [ ...
    cbLeft, ...
    0.19, ...
    cbWidth, ...
    0.035];

pdfFile = 'Ex3-Morse-muUNC_heatmaps_with_Nominal_drift_streamslice.pdf';

drawnow;

exportgraphics(fig, ...
    pdfFile, ...
    'ContentType','vector', ...
    'Padding','tight');

%% Heat maps of \mu_t <F_t[\mu_t],u_t>

fig = figure('Color','w','Position',[50 100 2600 500]);

nPanelsDot = numel(plotIdx);

axList = gobjects(nPanelsDot,1);
cbList = gobjects(nPanelsDot,1);


for pidx = 1:nPanelsDot

    ax = axes(fig,'Position',[ ...
        leftMargin + (pidx-1)*(panelW + gap) + 0.10*panelW, ...
        bottomMargin + 0.10*panelH, ...
        0.6*panelW, ...
        0.9*panelH]);

    axList(pidx) = ax;

    n = plotIdx(pidx);

    U = mu(:,:,n);

    logphi_INT  = log(max(varphi(:,:,n),rhoFloor));
    dlogphi_INT = centered_Dv_2D(logphi_INT,dv);
    uCtrl       = sigma2*dlogphi_INT;

    [~,~,FINT] = morse_force_coefficients(U,Kmat,dx,dv);


    DotShow = U(ixPlot,ivPlot) .* ...
              FINT(ixPlot,ivPlot) .* ...
              uCtrl(ixPlot,ivPlot);

    Ashow = DotShow';

   L = isfinite(Ashow);
  absA = abs(Ashow(isfinite(Ashow)));

if isempty(absA) || max(absA) < 1e-14
    localAlignmentScale = 1e-14;
else
    absA = sort(absA);
    idxScale = max(1,ceil(0.97*numel(absA)));
    localAlignmentScale = absA(idxScale);

    if localAlignmentScale < 1e-14
        localAlignmentScale = max(absA);
    end
end


    %% 
    hImg = imagesc(ax, ...
        x(ixPlot), ...
        v(ivPlot), ...
        Ashow);

    axis(ax,'xy');

    colormap(ax,turbo(256));

   
    clim(ax,[-localAlignmentScale localAlignmentScale]);


    hImg.AlphaData = ones(size(Ashow));

    xlim(ax,[xPlotMin xPlotMax]);
    ylim(ax,[vPlotMin vPlotMax]);


    set(ax, ...
        'FontSize',18, ...
        'TickLabelInterpreter','latex', ...
        'XTick',[-2 0 2], ...
        'YTick',[-3 0 3], ...
        'TickDir','out', ...
        'XColor','k', ...
        'YColor','k');

    if pidx == 1
        box(ax,'on');

        ax.XTickLabelMode = 'auto';
        ax.YTickLabelMode = 'auto';

        text(ax,0.50,1.04,'$x$', ...
            'Units','normalized', ...
            'Interpreter','latex', ...
            'FontSize',27, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', ...
            'Clipping','off');

        ylabel(ax,'$v$', ...
            'Interpreter','latex', ...
            'FontSize',27, ...
            'Rotation',0);

        ax.YLabel.Units = 'normalized';
        ax.YLabel.Position = [-0.18 0.50 0];
        ax.YLabel.HorizontalAlignment = 'center';
        ax.YLabel.VerticalAlignment = 'middle';

    else

        box(ax,'off');

        set(ax, ...
            'XTick',[], ...
            'YTick',[], ...
            'XTickLabel',[], ...
            'YTickLabel',[], ...
            'TickLength',[0 0], ...
            'XColor','none', ...
            'YColor','none');

    end

    axPos = ax.Position;

    cb = colorbar(ax,'southoutside');
    cbList(pidx) = cb;

    cb.FontSize = 14;
    cb.TickLabelInterpreter = 'latex';
    cb.TickDirection = 'out';

    % Colorbar now directly represents the actual plotted values
  
    cb.Ticks = [-localAlignmentScale 0 localAlignmentScale];
    cb.TickLabels = { ...
        sprintf('%.1f',-localAlignmentScale), ...
        '0.0', ...
        sprintf('%.1f', localAlignmentScale)};
       

    if pidx == 1

        cb.Label.String = ...
            '$\mu_t\,\langle g_t[\rho_t],u_t\rangle$';
        cb.Label.Interpreter = 'latex';
        cb.Label.FontSize = 22;
        cb.Label.Rotation = 0;
        cb.Label.HorizontalAlignment = 'center';

    else

        cb.Label.String = '';

    end

    ax.Position = axPos;


    cbWidth  = 0.82*axPos(3);
    cbHeight = 0.035;
    cbLeft   = axPos(1) + 0.5*(axPos(3)-cbWidth);
    cbBottom = 0.19;

    cb.Position = [cbLeft, cbBottom, cbWidth, cbHeight];

end

drawnow;

exportgraphics(fig, ...
    'Ex3-Morse-mu_force_dot_u_individual_colorbars.pdf', ...
    'Resolution',300);
%%
xPlotMin = -2; 
xPlotMax =  2;
vPlotMin = -3;
vPlotMax =  3;

ixPlot = find(x >= xPlotMin & x <= xPlotMax);
ivPlot = find(v >= vPlotMin & v <= vPlotMax);

%% 
figRho3D = figure('Color','w','Position',[100 100 1100 780]);
ax = axes(figRho3D);
hold(ax,'on');
nCurves = 9;

plotTimesMarg = linspace(0,1,nCurves);
plotIdxMarg   = round(plotTimesMarg*Nt) + 1;


rhoINT_all = squeeze(sum(mu,2))*dv;   % Nx x (Nt+1)
mINT_all  = squeeze(sum(mu,1))*dx;   % Nv x (Nt+1)

rhoGivenIn  = sum(mu_in,2)*dv;
rhoGivenFin = sum(mu_fin,2)*dv;

mGivenIn  = sum(mu_in,1)*dx;
mGivenFin = sum(mu_fin,1)*dx;


colororder(ax,'default');


for p = 1:nCurves

    n  = plotIdxMarg(p);
    tt = plotTimesMarg(p);

    plot3(ax, ...
        x(ixPlot), ...
        tt*ones(size(x(ixPlot))), ...
        rhoINT_all(ixPlot,n), ...
        'LineWidth',4);
end


plot3(ax, ...
    x(ixPlot), ...
    zeros(size(x(ixPlot))), ...
    rhoGivenIn(ixPlot), ...
    'k--', ...
    'LineWidth',4);


plot3(ax, ...
    x(ixPlot), ...
    ones(size(x(ixPlot))), ...
    rhoGivenFin(ixPlot), ...
    'k--',...
    'LineWidth',4);

hold(ax,'off');

grid(ax,'off');
box(ax,'on');

xlim(ax,[xPlotMin xPlotMax]);
ylim(ax,[0 1]);
pbaspect([1 2.5 1])
set(ax, ...
    'FontSize',25, ...
    'TickLabelInterpreter','latex', ...
    'LineWidth',1);

xlabel(ax,'$x$', ...
    'Interpreter','latex', ...
    'FontSize',35);

ylabel(ax,'$t$', ...
    'Interpreter','latex', ...
    'FontSize',35);

zlabel(ax,'$\rho_t(x)$', ...
    'Interpreter','latex', ...
    'FontSize',35);

view(ax,[-60 15]);

exportgraphics(figRho3D, ...
    'Ex3-Morse-rho_interacting_selected_times.pdf', 'padding','tight',...
    'Resolution',300);
%% Velocity marginals

figXi3D = figure('Color','w','Position',[100 100 1100 780]);
ax = axes(figXi3D);
hold(ax,'on');


colororder(ax,'default');


for p = 1:nCurves

    n  = plotIdxMarg(p);
    tt = plotTimesMarg(p);

    plot3(ax, ...
        v(ivPlot), ...
        tt*ones(size(v(ivPlot))), ...
        mINT_all(ivPlot,n), ...
        'LineWidth',4);
end


plot3(ax, ...
    v(ivPlot), ...
    zeros(size(v(ivPlot))), ...
    mGivenIn(ivPlot), ...
    'k--', ...
    'LineWidth',4);


plot3(ax, ...
    v(ivPlot), ...
    ones(size(v(ivPlot))), ...
    mGivenFin(ivPlot), ...
    'k--', ...
    'LineWidth',4);

hold(ax,'off');

grid(ax,'off');
box(ax,'on');

xlim(ax,[vPlotMin vPlotMax]);
ylim(ax,[0 1]);
pbaspect([1 2.5 1])
set(ax, ...
    'FontSize',25, ...
    'TickLabelInterpreter','latex', ...
    'LineWidth',1);

xlabel(ax,'$v$', ...
    'Interpreter','latex', ...
    'FontSize',35);

ylabel(ax,'$t$', ...
    'Interpreter','latex', ...
    'FontSize',35);

zlabel(ax,'$\xi_t(v)$', ...
    'Interpreter','latex', ...
    'FontSize',35);

view(ax,[-60 15]);

exportgraphics(figXi3D, ...
    'Ex3-Morse-velocity_marginal_interacting_selected_times.pdf',  "Padding","tight", ...
    'Resolution',300);
%%

ixP = 1:Nx;
ivP = 2:Nv-1;

[Xgrid,Vgrid] = ndgrid(x,v);

plotTimes = linspace(0,1,5);
plotIdx = round(plotTimes*Nt) + 1;

xForceMin = -3.5;
xForceMax = 3.5;
ixForce = find(x >= xForceMin & x <= xForceMax);

forceSliceVelocity = 0;
[~,ivForceSlice] = min(abs(v-forceSliceVelocity));
Fplot = zeros(numel(ixForce),numel(plotIdx));

timeWeights = ones(Nt+1,1);
timeWeights(1) = 0.5;
timeWeights(end) = 0.5;

J_INT_time = zeros(Nt+1,1);
J_SB_time  = zeros(Nt+1,1);
J_REF_time = zeros(Nt+1,1);


ForcePower_INT   = zeros(Nt+1,1);
ControlPower_INT = zeros(Nt+1,1);
TotalPower_INT   = zeros(Nt+1,1);
ControlPower_SB  = zeros(Nt+1,1);

for n = 1:Nt+1

    U = mu(:,:,n);

    logphi_INT = log(max(varphi(:,:,n),rhoFloor));
    dlogphi_INT = centered_Dv_2D(logphi_INT,dv);
    u_INT = sigma2*dlogphi_INT;

    [~,~,FINT] = morse_force_coefficients(U,Kmat,dx,dv);

    [isPlotTime,pidx] = ismember(n,plotIdx);
    if isPlotTime
        Fplot(:,pidx) = FINT(ixForce,ivForceSlice);
    end

    J_INT_time(n) = ...
    (0.5/sigma2)* ...
    sum((u_INT(ixP,ivP).^2).*U(ixP,ivP),'all')*dx*dv;

    mx = sum(Xgrid.*U,'all')*dx*dv;
    mv = sum(Vgrid.*U,'all')*dx*dv;

    Up = U(ixP,ivP);
    Xp = Xgrid(ixP,ivP);
    Vp = Vgrid(ixP,ivP);
    Fp = FINT(ixP,ivP);
    uCtrlP = u_INT(ixP,ivP);


    ForcePower_INT(n) = sum((Vp.*Fp).*Up,'all')*dx*dv;
    ControlPower_INT(n) = ...
    sum((Vp.*uCtrlP).*Up,'all')*dx*dv;    
    TotalPower_INT(n) = ForcePower_INT(n) + ControlPower_INT(n);

    USB = muSB(:,:,n);

    logphi_SB = log(max(varphiSB(:,:,n),rhoFloor));
    dlogphi_SB = centered_Dv_2D(logphi_SB,dv);
    u_SB = sigma2*dlogphi_SB;

    [~,~,FSB] = morse_force_coefficients(USB,Kmat,dx,dv);

    J_SB_time(n) = ...
    (0.5/sigma2)* ...
    sum((u_SB(ixP,ivP).^2).*USB(ixP,ivP),'all')*dx*dv;

    u_REF = u_SB - FSB;
    J_REF_time(n) = ...
    (0.5/sigma2)* ...
    sum((u_REF(ixP,ivP).^2).*USB(ixP,ivP),'all')*dx*dv;

    mxSB = sum(Xgrid.*USB,'all')*dx*dv;
    mvSB = sum(Vgrid.*USB,'all')*dx*dv;


    USBp = USB(ixP,ivP);
    VSBp = Vgrid(ixP,ivP);
    uCtrlSBp = u_SB(ixP,ivP);

    ControlPower_SB(n) = ...
    sum((VSBp.*uCtrlSBp).*USBp,'all')*dx*dv;

    
end

J_INT = sum(timeWeights.*J_INT_time)*dt;
J_SB  = sum(timeWeights.*J_SB_time)*dt;
J_REF = sum(timeWeights.*J_REF_time)*dt;

fprintf('Morse-interacting candidate cost              = %.8e\n',J_INT);
fprintf('Cost of non-interacting bridge              = %.8e\n',J_SB);
fprintf('Cost of feasible interacting reference (baseline)      = %.8e\n',J_REF);
fprintf('Candidate minus feasible reference          = %.8e\n',J_INT - J_REF);

fig = figure;

plot(t,J_INT_time,'LineWidth',5); 
hold on;

plot(t,J_SB_time,'-.','LineWidth',5);
plot(t,J_REF_time,'k:','LineWidth',5);
grid off;
set(ax, ...
    'FontSize',25, ...
    'TickLabelInterpreter','latex');
xlabel('$t$','Interpreter','latex','FontSize',35);
ylabel('Instantaneous Cost', ...
    'Interpreter','latex', ...
    'FontSize',35);legend('Interacting','Non-interacting', ...
    'Baseline','Location','northeast','Interpreter','latex', ...
    'FontSize',30);
legend boxoff;
pbaspect([2 1 1])

exportgraphics(fig, 'Ex3-Morse-cost.pdf', ...
    'Resolution',300);

%% Local functions

function [mu_sb,varphi_sb,hatvarphi_sb,errSB,lastIter,errPhi1,errHat1] = strang_SB_mu_endpoint( ...
    mu_in,mu_fin,dv,dt,D,Nt,AvHalfIntDec,AvHalfAdjIntDec,T0, ...
    bcValVarphi,bcValHat,bcMu,posTol,thr,tolSB,maxIter)

    [Nx,Nv] = size(mu_in);
    varphi1 = impose_phase_boundary(ones(Nx,Nv),bcValVarphi);
    errSB   = nan(maxIter,1);
    errPhi1 = nan(maxIter,1);
    errHat1 = nan(maxIter,1);
    lastIter = maxIter;


hat1_old = [];

for iter = 1:maxIter
    u = varphi1;
    for tt = Nt+1:-1:2
        u = strang_adj_interacting(u,AvHalfAdjIntDec,T0,zeros(Nx,1),dt,D,dv,bcValVarphi,posTol,'SB adjoint');
    end

    hat0 = divide_on_interior(mu_in,u,bcValHat,posTol,'SB hat0');

    uhat = hat0;
    for tt = 1:Nt
        uhat = strang_interacting(uhat,AvHalfIntDec,T0,zeros(Nx,1),dt,D,dv,bcValHat,posTol,'SB forward');
    end

    varphi1_next = divide_on_interior(mu_fin,uhat,bcValVarphi,posTol,'SB varphi1');

    errPhi1(iter) = hilbert_metric(varphi1,varphi1_next,thr,2);

    if isempty(hat1_old)
        errHat1(iter) = Inf;
    else
        errHat1(iter) = hilbert_metric(hat1_old,uhat,thr,2);
    end

    errSB(iter) = max(errPhi1(iter),errHat1(iter));

    varphi1 = varphi1_next;
    hat1_old = uhat;

    if errSB(iter) < tolSB
        lastIter = iter;
        break;
    end
end

    varphi_sb = bcValVarphi*ones(Nx,Nv,Nt+1);
    varphi_sb(:,:,end) = impose_phase_boundary(varphi1,bcValVarphi);
    for tt = Nt+1:-1:2
        varphi_sb(:,:,tt-1) = strang_adj_interacting(varphi_sb(:,:,tt),AvHalfAdjIntDec,T0,zeros(Nx,1),dt,D,dv,bcValVarphi,posTol,'SB traj adjoint');
    end

    hatvarphi_sb = bcValHat*ones(Nx,Nv,Nt+1);
    hatvarphi_sb(:,:,1) = divide_on_interior(mu_in,varphi_sb(:,:,1),bcValHat,posTol,'SB traj hat0');
    for tt = 1:Nt
        hatvarphi_sb(:,:,tt+1) = strang_interacting(hatvarphi_sb(:,:,tt),AvHalfIntDec,T0,zeros(Nx,1),dt,D,dv,bcValHat,posTol,'SB traj forward');
    end

    mu_sb = bcMu*ones(Nx,Nv,Nt+1);
    for tt = 1:Nt+1
        mu_sb(:,:,tt) = reconstruct_density(varphi_sb(:,:,tt),hatvarphi_sb(:,:,tt),bcMu,posTol,sprintf('SB mu tt=%d',tt));
    end
end

function Uout = strang_interacting(U,AvHalfIntDec,T,G,dt,D,dv,bcVal,posTol,name)
    U = impose_phase_boundary(U,bcVal);
    require_positive_phase(U,[name ' input'],posTol);

    React = exp(-0.5*dt*G);
    U = multiply_phase(U,React);
    U = implicit_v(U,AvHalfIntDec,0.5*dt,D,dv,bcVal,posTol,[name ' diffusion half 1']);
    U = reshape(T*U(:),size(U));
    U = impose_phase_boundary(U,bcVal);
    require_positive_phase(U,[name ' transport'],posTol);
    U = implicit_v(U,AvHalfIntDec,0.5*dt,D,dv,bcVal,posTol,[name ' diffusion half 2']);
    Uout = multiply_phase(U,React);
    Uout = impose_phase_boundary(Uout,bcVal);
    require_positive_phase(Uout,[name ' output'],posTol);
end

function Uprev = strang_adj_interacting(U,AvHalfAdjIntDec,T,G,dt,D,dv,bcVal,posTol,name)
    U = impose_phase_boundary(U,bcVal);
    require_positive_phase(U,[name ' input'],posTol);

    React = exp(-0.5*dt*G);
    Y = multiply_phase(U,React);
    Y = implicit_v(Y,AvHalfAdjIntDec,0.5*dt,D,dv,bcVal,posTol,[name ' adj diffusion half 1']);
    Y = reshape(T.'*Y(:),size(U));
    Y = impose_phase_boundary(Y,bcVal);
    require_positive_phase(Y,[name ' adj transport'],posTol);
    Y = implicit_v(Y,AvHalfAdjIntDec,0.5*dt,D,dv,bcVal,posTol,[name ' adj diffusion half 2']);
    Uprev = multiply_phase(Y,React);
    Uprev = impose_phase_boundary(Uprev,bcVal);
    require_positive_phase(Uprev,[name ' output'],posTol);
end

function U = multiply_phase(U,R)
    if isvector(R) && numel(R) == size(U,1)
        U = U.*R(:);
    elseif isequal(size(R),size(U))
        U = U.*R;
    else
        error('Reaction multiplier has incompatible size.');
    end
end

function Uout = implicit_v(rhs,AvIntDec,dt,D,dv,bcVal,posTol,name)
    % Velocity diffusion with periodic x and Dirichlet/floored velocity boundary.
    % The solve is performed independently for every x-row.
    [Nx,Nv] = size(rhs);
    rhs = impose_phase_boundary(rhs,bcVal);
    r = dt*D/(dv^2);

    B = rhs(:,2:Nv-1);
    B(:,1) = B(:,1) + r*bcVal;
    B(:,end) = B(:,end) + r*bcVal;

    Uout = bcVal*ones(Nx,Nv);
    Uout(:,2:Nv-1) = (AvIntDec\B.').';
    Uout = impose_phase_boundary(Uout,bcVal);
    require_positive_phase(Uout,name,posTol);
end

function AvInt = diffusion_v_matrix(Nv,dt,D,dv)
    n = Nv-2;
    evec = ones(n,1);
    LvInt = spdiags([evec -2*evec evec],[-1 0 1],n,n)/(dv^2);
    AvInt = speye(n) - dt*D*LvInt;
end

function U = impose_phase_boundary(U,bcVal)
    % Periodic boundary in x: no x-row is overwritten.
    % Velocity boundary is kept at bcVal.
    U(:,1) = bcVal;
    U(:,end) = bcVal;
end

function require_positive_phase(U,name,posTol)
    if any(~isfinite(U(:)))
        error('%s has nonfinite entries.',name);
    end

    % Periodic x: all x rows are physical.  Only velocity boundary columns
    % are excluded from the positivity check.
    interior = U(:,2:end-1);
    if any(interior(:) <= posTol)
        [val,idx0] = min(interior(:));
        error('%s lost positivity away from v-boundary: min=%e at linear index %d.',name,val,idx0);
    end
end

function Kmat = morse_kernel_matrix(S,Lx,C_R,C_A,ell_R,ell_A)
    % Minimum-image Morse force kernel on the periodic spatial grid.
    R = abs(S);
    Kmat = (C_R/ell_R)*exp(-R/ell_R) ...
         - (C_A/ell_A)*exp(-R/ell_A);
    Kmat = Kmat.*sign(S);

    % Resolve the antipodal minimum-image tie and enforce odd symmetry.
    Kmat(abs(R-0.5*Lx) <= 100*eps(max(1,Lx))) = 0;
    Kmat = 0.5*(Kmat-Kmat.');
    Kmat(1:size(Kmat,1)+1:end) = 0;
end

function [rho,g,F] = morse_force_coefficients(U,Kmat,dx,dv)
    % rho(x) = int U(x,v)dv and g[rho](x) = int K_M(x-x~)rho(x~)dx~.
    rho = sum(U,2)*dv;
    g = (Kmat*rho)*dx;

    if nargout >= 3
        F = g*ones(1,size(U,2));
    end
end

function [T,clipMask] = morse_pullback(x,v,g,dt)
    % Semi-Lagrangian pullback for the phase-space drift (v,g(x)).
    Nx = numel(x);
    Nv = numel(v);
    dx = x(2)-x(1);
    dv = v(2)-v(1);
    Lx = Nx*dx;
    xLeft = x(1);
    N = Nx*Nv;

    clipMask = false(Nx,Nv);

    rows = zeros(4*N,1);
    cols = zeros(4*N,1);
    vals = zeros(4*N,1);
    kk = 0;

    for iv = 1:Nv
        for ix = 1:Nx
            row = ix + (iv-1)*Nx;

            if iv == 1 || iv == Nv
                kk = kk + 1;
                rows(kk) = row;
                cols(kk) = row;
                vals(kk) = 1;
                continue;
            end

            h = -dt;
            x1 = x(ix);
            v1 = v(iv);

            g1 = g(ix);
            xMid = wrap_periodic_scalar(x1 + 0.5*h*v1,xLeft,Lx);
            vMid = v1 + 0.5*h*g1;
            gMid = periodic_interp_scalar(x,g,xMid);

            x0 = wrap_periodic_scalar(x1 + h*vMid,xLeft,Lx);
            v0Raw = v1 + h*gMid;
            v0 = min(max(v0Raw,v(1)),v(end));

            if v0Raw ~= v0
                clipMask(ix,iv) = true;
            end

            sx = (x0-xLeft)/dx + 1;
            ix0 = floor(sx);
            if ix0 < 1
                ix0 = 1;
            elseif ix0 > Nx
                ix0 = Nx;
            end
            ix1 = ix0 + 1;
            if ix1 > Nx
                ix1 = 1;
            end
            ax = sx - ix0;

            sv = (v0-v(1))/dv + 1;
            iv0 = max(1,min(Nv-1,floor(sv)));
            av = sv - iv0;

            w00 = (1-ax)*(1-av);
            w10 = ax*(1-av);
            w01 = (1-ax)*av;
            w11 = ax*av;

            col00 = ix0 + (iv0-1)*Nx;
            col10 = ix1 + (iv0-1)*Nx;
            col01 = ix0 + iv0*Nx;
            col11 = ix1 + iv0*Nx;

            kk = kk + 1; rows(kk)=row; cols(kk)=col00; vals(kk)=w00;
            kk = kk + 1; rows(kk)=row; cols(kk)=col10; vals(kk)=w10;
            kk = kk + 1; rows(kk)=row; cols(kk)=col01; vals(kk)=w01;
            kk = kk + 1; rows(kk)=row; cols(kk)=col11; vals(kk)=w11;
        end
    end

    T = sparse(rows(1:kk),cols(1:kk),vals(1:kk),N,N);
end

function S = periodic_distance_matrix(x)
    % Signed shortest periodic displacement matrix S(i,j)=x_i-x_j on a torus.
    Nx = numel(x);
    dx = x(2)-x(1);
    Lx = Nx*dx;
    S = x - x.';
    S = mod(S + 0.5*Lx,Lx) - 0.5*Lx;
end

function xw = wrap_periodic_scalar(xq,xLeft,Lx)
    xw = mod(xq - xLeft,Lx) + xLeft;
end

function yq = periodic_interp_scalar(x,y,xq)
    % Linear interpolation on the periodic grid x, where x(end)+dx is x(1).
    Nx = numel(x);
    dx = x(2)-x(1);
    Lx = Nx*dx;
    xLeft = x(1);

    xq = wrap_periodic_scalar(xq,xLeft,Lx);
    sx = (xq-xLeft)/dx + 1;
    i0 = floor(sx);
    if i0 < 1
        i0 = 1;
    elseif i0 > Nx
        i0 = Nx;
    end
    i1 = i0 + 1;
    if i1 > Nx
        i1 = 1;
    end
    a = sx - i0;
    yq = (1-a)*y(i0) + a*y(i1);
end


function U = normalize2d_vboundary(U,dx,dv,bcVal,posTol,name)
    % Normalize a phase-space density with periodic x and fixed v-boundary.
    U = impose_phase_boundary(U,bcVal);
    require_positive_phase(U,name,posTol);

    boundaryMask = false(size(U));
    boundaryMask(:,1) = true;
    boundaryMask(:,end) = true;

    interiorMass = sum(U(:,2:end-1),'all')*dx*dv;
    boundaryMass = sum(U(boundaryMask),'all')*dx*dv;
    targetInteriorMass = 1 - boundaryMass;

    if targetInteriorMass <= 0 || interiorMass <= 0
        error('%s has bad v-boundary normalization.',name);
    end

    U(:,2:end-1) = U(:,2:end-1)*(targetInteriorMass/interiorMass);
    U = impose_phase_boundary(U,bcVal);
end

function U = reconstruct_density(varphi,hatvarphi,bcMu,posTol,name)
    [Nx,Nv] = size(varphi);
    if ~isequal(size(hatvarphi),[Nx,Nv])
        error('%s has incompatible factor sizes.',name);
    end

    require_positive_phase(varphi,[name ' varphi'],posTol);
    require_positive_phase(hatvarphi,[name ' hatvarphi'],posTol);

    U = bcMu*ones(Nx,Nv);
    iv = 2:Nv-1;
    U(:,iv) = varphi(:,iv).*hatvarphi(:,iv);
    U = impose_phase_boundary(U,bcMu);
    require_positive_phase(U,name,posTol);
end

function U = damp_density(cand,old,theta,bcMu,posTol,name)
    [Nx,Nv] = size(cand);
    if ~isequal(size(old),[Nx,Nv])
        error('%s has incompatible density sizes.',name);
    end

    U = bcMu*ones(Nx,Nv);
    iv = 2:Nv-1;
    U(:,iv) = theta*cand(:,iv) + (1-theta)*old(:,iv);
    U = impose_phase_boundary(U,bcMu);
    require_positive_phase(U,name,posTol);
end

function dUdv = centered_Dv_2D(U,dv)
    [Nx,Nv] = size(U);
    dUdv = zeros(Nx,Nv);
    dUdv(:,2:Nv-1) = (U(:,3:Nv)-U(:,1:Nv-2))/(2*dv);
end

function U = divide_on_interior(num,den,bcVal,posTol,name)
    [Nx,Nv] = size(num);
    U = bcVal*ones(Nx,Nv);
    idxV = 2:Nv-1;
    denInt = den(:,idxV);
    if any(~isfinite(denInt(:))) || any(denInt(:) <= 0)
        error('%s has nonpositive denominator away from v-boundary.',name);
    end
    U(:,idxV) = num(:,idxV)./denInt;
    U = impose_phase_boundary(U,bcVal);
    require_positive_phase(U,name,posTol);
end

function d = hilbert_metric(U,V,thr,clipH)
    if nargin < 4
        clipH = 0;
    end

    if isvector(U)
        U = U(:);
        V = V(:);
    else
        % Periodic x: do not remove x-endpoints.  Clip only the artificial
        % velocity boundary columns if requested.
        if clipH > 0 && size(U,2) > 2*clipH
            U = U(:,(1+clipH):(end-clipH));
            V = V(:,(1+clipH):(end-clipH));
        end
    end

    u = U(:);
    v = V(:);
    cond = isfinite(u) & isfinite(v) & (u > thr) & (v > thr);
    if nnz(cond) < 3
        cond = isfinite(u) & isfinite(v) & (u > 0) & (v > 0);
    end
    if nnz(cond) == 0
        d = Inf;
        return;
    end
    r = u(cond)./v(cond);
    d = log(max(r(:))/min(r(:)));
end

