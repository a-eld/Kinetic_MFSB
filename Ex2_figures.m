close all;

resultsFile = fullfile( ...
    pwd, ...
    'saved_results', ...
    'Ex2_CuckerSmale_partial_results.mat');

load(resultsFile);

sigma2 = params.sigma2;
sigma  = sqrt(sigma2);
D      = params.D;

Nx = params.Nx;
Nv = params.Nv;
Nt = params.Nt;

dx = params.dx;
dv = params.dv;
dt = params.dt;

bcMu   = params.bcMu;
posTol = params.posTol;

lastOuter   = diagnostics.lastOuter;
massErrPlot = diagnostics.massErr;


%%
rhoFloor = 1E-14;
plotTimes = linspace(0,1,5);
plotIdx = round(plotTimes*Nt) + 1;

xPlotMin = -2; 
xPlotMax =  2;
vPlotMin = -2.8;
vPlotMax =  2.8;

ixPlot = find(x >= xPlotMin & x <= xPlotMax);
ivPlot = find(v >= vPlotMin & v <= vPlotMax);

[X0,V0] = meshgrid(x(ixPlot),v(ivPlot));

%% Nominal Cucker--Smale evolution from the prior muQ0

dtHalf = 0.5*dt;

AvHalfInt = diffusion_v_matrix(Nv,dtHalf,D,dv);
AvHalfIntDec = decomposition(AvHalfInt,'lu');

%% Nominal Cucker--Smale evolution

muNom = bcMu*ones(Nx,Nv,Nt+1);

if ndims(muQ0) == 3
    muNom(:,:,1) = impose_phase_boundary(muQ0(:,:,1),bcMu);
else
    muNom(:,:,1) = impose_phase_boundary(muQ0,bcMu);
end

massNom = zeros(Nt+1,1);
massNom(1) = sum(muNom(:,:,1),'all')*dx*dv;

for n = 1:Nt

    U0 = muNom(:,:,n);

    [~,~,alignRate0,weightedMomentum0] = ...
        cs_force_coefficients(U0,v,Wmat,dx,dv);

    F0 = weightedMomentum0 ...
       - alignRate0.*v(:).';


    Upred = implicit_v( ...
        U0,AvHalfIntDec,dtHalf,D,dv,bcMu,posTol, ...
        sprintf('Nominal predictor diffusion half 1 n=%d',n));

    Upred = cs_conservative_transport_rk2( ...
        Upred,F0,v,dx,dv,dt);

    Upred = implicit_v( ...
        Upred,AvHalfIntDec,dtHalf,D,dv,bcMu,posTol, ...
        sprintf('Nominal predictor diffusion half 2 n=%d',n));

    [~,~,alignRate1,weightedMomentum1] = ...
        cs_force_coefficients(Upred,v,Wmat,dx,dv);

    alignRateHalf = 0.5*(alignRate0 + alignRate1);

    weightedMomentumHalf = ...
        0.5*(weightedMomentum0 + weightedMomentum1);

    Fhalf = weightedMomentumHalf ...
          - alignRateHalf.*v(:).';

    
    U1 = implicit_v( ...
        U0,AvHalfIntDec,dtHalf,D,dv,bcMu,posTol, ...
        sprintf('Nominal corrected diffusion half 1 n=%d',n));

    U1 = cs_conservative_transport_rk2( ...
        U1,Fhalf,v,dx,dv,dt);

    U1 = implicit_v( ...
        U1,AvHalfIntDec,dtHalf,D,dv,bcMu,posTol, ...
        sprintf('Nominal corrected diffusion half 2 n=%d',n));

 

    muNom(:,:,n+1) = U1;

    massNom(n+1) = sum(U1,'all')*dx*dv;

 
end

fprintf('\nNominal Cucker--Smale evolution from prior muQ0:\n');
fprintf('maximum mass error = %.3e\n', ...
    max(abs(massNom-massNom(1))));
fprintf('final mass          = %.12f\n',massNom(end));
%%

zMax = 0;
cMin = Inf;
cMax = -Inf;

for pidx = 1:numel(plotIdx)
    n = plotIdx(pidx);

    Z0 = mu(ixPlot,ivPlot,n)';
   
    zMax = max(zMax,max(Z0,[],'all'));

    cMin = min(cMin,min(Z0,[],'all'));
    cMax = max(cMax,max(Z0,[],'all'));
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

    Z0 = mu(ixPlot,ivPlot,n)';

    [Xq,Vq] = meshgrid( ...
        linspace(x(ixPlot(1)),x(ixPlot(end)),200), ...
        linspace(v(ivPlot(1)),v(ivPlot(end)),300));

    Z_dense = interp2( ...
        X0,V0,Z0, ...
        Xq,Vq, ...
        'cubic');
    s = surf(ax,Xq,Vq,Z_dense, ...
        'EdgeColor','none', ...
        'FaceColor','interp', ...
        'FaceLighting','gouraud');


    shading(ax,'interp');
    colormap(fig,turbo);

    clim(ax,[cMin,1.9]);

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
        'YTick',[-2 0 2], ...
        'ZTick',[], ...
        'TickDir','out');

   
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


ax.XAxis.TickLabelGapOffset = -5;
ax.YAxis.TickLabelGapOffset = -5;


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
exportgraphics(fig,'Ex2-phase_space_surfaces.pdf','Resolution',300);


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

    logphi_INT = log(max(varphi(:,:,n),rhoFloor));
    dlogphi_INT = centered_Dv_2D(logphi_INT,dv);
    u_INT = sigma2*dlogphi_INT;

    [~,~,~,~,FINT] = ...
        cs_force_coefficients(U,v,Wmat,dx,dv);

    imagesc(ax, ...
        x(ixPlot), ...
        v(ivPlot), ...
        U(ixPlot,ivPlot)');

    axis(ax,'xy');

    colormap(fig,turbo);
    clim(ax,[cMin,2]);

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

    axis(ax,'xy');
    box(ax,'on');

    
    set(ax, ...
        'FontSize',18, ...
        'TickLabelInterpreter','latex', ...
        'XTick',[-2 0 2], ...
        'YTick',[-2 0 2], ...
        'TickDir','out', ...
        'XColor','k', ...
        'YColor','k');

   
    if pidx == 1

      
        axis(ax,'on');
        box(ax,'on');

        set(ax, ...
            'FontSize',18, ...
            'TickLabelInterpreter','latex', ...
            'XTick',[-2 0 2], ...
            'YTick',[-2 0 2], ...
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
cb.Ticks = unique([[0,0.5,1,1.5,2],colorLimits(1),colorLimits(2)]);
cb.TickLabels = arrayfun( ...
    @(value) sprintf('%.1f',value), ...
    cb.Ticks, ...
    'UniformOutput',false);


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

pdfFile = 'Ex2-mu_heatmaps_with_drift_streamslice.pdf';

drawnow;

exportgraphics(fig, ...
    pdfFile, ...
    'ContentType','vector', ...
    'Padding','tight');


%% Heat maps of \mu_t <F_t[\mu_t],u_t> 
nPanelsAlignment = numel(plotIdx);

alignmentPlot = cell(nPanelsAlignment,1);
densityPlot   = cell(nPanelsAlignment,1);

allAlignmentValues = [];

for pidx = 1:nPanelsAlignment

    n = plotIdx(pidx);

    U = mu(:,:,n);

    logphi_INT  = log(max(varphi(:,:,n),rhoFloor));
    dlogphi_INT = centered_Dv_2D(logphi_INT,dv);
    u_INT       = sigma2*dlogphi_INT;

  
    aCtrl = u_INT;

   
    [~,~,~,~,FINT] = cs_force_coefficients(U,v,Wmat,dx,dv);

   
    alignmentDensity = U .* FINT .* aCtrl;

    alignmentPlot{pidx} = alignmentDensity(ixPlot,ivPlot)';
    densityPlot{pidx}   = U(ixPlot,ivPlot)';

    allAlignmentValues = [allAlignmentValues; alignmentPlot{pidx}(:)]; %#ok<AGROW>
end


%% 
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


    [~,~,~,~,FINT] = cs_force_coefficients(U,v,Wmat,dx,dv);


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
        'YTick',[-2 0 2], ...
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

  
    cb.Ticks = [-localAlignmentScale 0 localAlignmentScale];
    cb.TickLabels = { ...
        sprintf('%.1f',-localAlignmentScale), ...
        '0.0', ...
        sprintf('%.1f', localAlignmentScale)};
       

    if pidx == 1

        cb.Label.String = ...
            '$\mu_t\,\langle f_t[\mu_t],u_t\rangle$';
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
    'Ex2-mu_force_dot_u_individual_colorbars.pdf', ...
    'Resolution',300);
%%

xPlotMin = -2; 
xPlotMax =  2;
vPlotMin = -2.8;
vPlotMax =  2.8;

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
mINT_all   = squeeze(sum(mu,1))*dx;   % Nv x (Nt+1)


rhoGivenIn  = rho_in;
rhoGivenFin = rho_fin;


if ndims(muQ0) == 3
    muPrior0 = muQ0(:,:,1);
else
    muPrior0 = muQ0;
end


mPrior0 = sum(muPrior0,1)*dx;
mPrior0 = mPrior0(:);


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
    'k--', ...
    'LineWidth',4);

hold(ax,'off');

grid(ax,'off');
box(ax,'on');

xlim(ax,[xPlotMin xPlotMax]);
ylim(ax,[0 1]);

pbaspect([1 2.5 1]);

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
    'Ex2-rho_interacting_selected_times.pdf', ...
    'Padding','tight', ...
    'Resolution',300);


%% 

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
    zeros(size(v(ivPlot)))-0.002, ...
    mPrior0(ivPlot), ...
    'k--', ...
    'LineWidth',4);

hold(ax,'off');

grid(ax,'off');
box(ax,'on');

xlim(ax,[vPlotMin vPlotMax]);
ylim(ax,[-0.002 1]);

pbaspect([1 2.5 1]);

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
    'Ex2-velocity_marginal_interacting_selected_times.pdf', ...
    'Padding','tight', ...
    'Resolution',300);
%%
ixP = 1:Nx;
ivP = 2:Nv-1;

[Xgrid,Vgrid] = ndgrid(x,v);

plotTimes = linspace(0,1,5);
plotIdx = round(plotTimes*Nt) + 1;

xForceMin = -3;
xForceMax = 3;
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

    [~,~,~,~,FINT] = cs_force_coefficients(U,v,Wmat,dx,dv);

    [isPlotTime,pidx] = ismember(n,plotIdx);
    if isPlotTime
        Fplot(:,pidx) = FINT(ixForce,ivForceSlice);
    end

    J_INT_time(n) = ...
        (1/(2*sigma2))* ...
        sum((u_INT(ixP,ivP).^2).*U(ixP,ivP),'all')*dx*dv;

    mx = sum(Xgrid.*U,'all')*dx*dv;
    mv = sum(Vgrid.*U,'all')*dx*dv;

    Up = U(ixP,ivP);
    Xp = Xgrid(ixP,ivP);
    Vp = Vgrid(ixP,ivP);
    Fp = FINT(ixP,ivP);
    aCtrlP = u_INT(ixP,ivP);

    ForcePower_INT(n) = ...
        sum((Vp.*Fp).*Up,'all')*dx*dv;

    ControlPower_INT(n) = ...
        sum((Vp.*aCtrlP).*Up,'all')*dx*dv;

    TotalPower_INT(n) = ...
        ForcePower_INT(n) + ControlPower_INT(n);

    USB = muSB(:,:,n);

    logphi_SB = log(max(varphiSB(:,:,n),rhoFloor));
    dlogphi_SB = centered_Dv_2D(logphi_SB,dv);
    u_SB = sigma2*dlogphi_SB;

    [~,~,~,~,FSB] = cs_force_coefficients(USB,v,Wmat,dx,dv);

    J_SB_time(n) = ...
        (1/(2*sigma2))* ...
        sum((u_SB(ixP,ivP).^2).*USB(ixP,ivP),'all')*dx*dv;

    u_REF = u_SB - FSB;

    J_REF_time(n) = ...
        (1/(2*sigma2))* ...
        sum((u_REF(ixP,ivP).^2).*USB(ixP,ivP),'all')*dx*dv;

    mxSB = sum(Xgrid.*USB,'all')*dx*dv;
    mvSB = sum(Vgrid.*USB,'all')*dx*dv;

    USBp = USB(ixP,ivP);
    VSBp = Vgrid(ixP,ivP);
    aCtrlSBp = u_SB(ixP,ivP);

    ControlPower_SB(n) = ...
        sum((VSBp.*aCtrlSBp).*USBp,'all')*dx*dv;

end

J_INT = sum(timeWeights.*J_INT_time)*dt;
J_SB  = sum(timeWeights.*J_SB_time)*dt;
J_REF = sum(timeWeights.*J_REF_time)*dt;

fprintf('Cucker--Smale candidate cost                = %.8e\n',J_INT);
fprintf('Cost of non-interacting bridge              = %.8e\n',J_SB);
fprintf('Cost of feasible interacting reference (baseline)     = %.8e\n',J_REF);
fprintf('Candidate minus feasible reference          = %.8e\n',J_INT - J_REF);

fig = figure;

plot(t,J_INT_time,'LineWidth',5); 
hold on;

plot(t,J_SB_time,'-.','LineWidth',5);
plot(t,J_REF_time,'k:','LineWidth',5);

grid off;

set(gca, ...
    'FontSize',25, ...
    'TickLabelInterpreter','latex');

xlabel('$t$', ...
    'Interpreter','latex', ...
    'FontSize',35);

ylabel('Control Energy', ...
    'Interpreter','latex', ...
    'FontSize',35);

legend('Interacting','Non-interacting', ...
    'Baseline', ...
    'Location','northeast', ...
    'Interpreter','latex', ...
    'FontSize',30);

legend boxoff;

pbaspect([2 1 1]);

exportgraphics(fig, ...
    'Ex2-cost.pdf', ...
    'Resolution',300);


%%

muINT0 = mu(:,:,1);
muSB0  = muSB(:,:,1);

H_INT = sum(muINT0(:,ivP).*log(muINT0(:,ivP)./muPrior0(:,ivP)),'all')*dx*dv;
H_SB  = sum(muSB0(:,ivP).*log(muSB0(:,ivP)./muPrior0(:,ivP)),'all')*dx*dv;

%% Full objective values

J_INT_full = H_INT + J_INT;
J_SB_full  = H_SB  + J_SB;
J_REF_full = H_SB  + J_REF;

fprintf('\nInitial KL divergence (interacting) = %.8e\n',H_INT);
fprintf('Initial KL divergence (classical)   = %.8e\n',H_SB);

fprintf('\nFull objective (interacting)        = %.8e\n',J_INT_full);
fprintf('Full objective (non-interacting)    = %.8e\n',J_SB_full);
fprintf('Full objective (baseline)           = %.8e\n',J_REF_full);

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

function [rho,momentum,alignRate,weightedMomentum,F] = cs_force_coefficients(U,v,Wmat,dx,dv)
    % Standard one-dimensional Cucker--Smale force:
    % f[U](x,v) = (w*m)(x) - (w*rho)(x)*v.
    rho = sum(U,2)*dv;
    momentum = sum(U.*v.',2)*dv;
    alignRate = (Wmat*rho)*dx;
    weightedMomentum = (Wmat*momentum)*dx;

    if nargout >= 5
        F = weightedMomentum*ones(1,numel(v)) - alignRate*v.';
    end
end

function [qSlope,qIntercept] = cs_q_coefficients(hatvarphi,dvarphi_dv,v,Wmat,dx,dv,iv)
    % q(x,v) = -int hatphi(x~,v~) w(x-x~) (v-v~)
    %                    d_{v~}phi(x~,v~) dx~dv~
    %          = qIntercept(x) - qSlope(x)*v.
    H = hatvarphi(:,iv).*dvarphi_dv(:,iv);
    source0 = sum(H,2)*dv;
    source1 = sum(H.*v(iv).',2)*dv;
    qSlope = (Wmat*source0)*dx;
    qIntercept = (Wmat*source1)*dx;
end

function G = cs_reaction_matrix(qSlope,qIntercept,alignRate,v)
    % The forward factor equation is
    % h_t = -v h_x - f h_v + D h_vv -(q+div_v f)h.
    % In 1D, div_v f = -alignRate, hence G=q-alignRate in exp(-dt*G).
    G = (qIntercept-alignRate)*ones(1,numel(v)) - qSlope*v.';
end

function [T,clipMask] = cs_affine_pullback(x,v,alignRate,weightedMomentum,dt)
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

            % Velocity boundary is fixed.  There is no x-boundary row because x is periodic.
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

            a1 = alignRate(ix);
            b1 = weightedMomentum(ix);
            F1 = b1 - a1*v1;

            xMid = wrap_periodic_scalar(x1 + 0.5*h*v1,xLeft,Lx);
            vMid = v1 + 0.5*h*F1;

            aMid = periodic_interp_scalar(x,alignRate,xMid);
            bMid = periodic_interp_scalar(x,weightedMomentum,xMid);
            FMid = bMid - aMid*vMid;

            x0 = wrap_periodic_scalar(x1 + h*vMid,xLeft,Lx);
            v0Raw = v1 + h*FMid;
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

function Uout = cs_conservative_transport_rk2( ...
    U,F,v,dx,dv,dt)

% Solve
%
% d_t U + d_x(v U) + d_v(F U) = 0
%
% with F held fixed during this transport substep.

%% CFL condition
cflRate = max(abs(v))/dx ...
    + max(abs(F),[],'all')/dv;

CFLtarget = 0.9;

nSub = max(1,ceil(dt*cflRate/CFLtarget));
h = dt/nSub;

Uout = U;

%% SSP-RK2 time stepping
for m = 1:nSub

    % Stage 1
    K1 = cs_conservative_transport_rhs( ...
        Uout,F,v,dx,dv);

    Ustar = Uout + h*K1;

    % Stage 2
    K2 = cs_conservative_transport_rhs( ...
        Ustar,F,v,dx,dv);

    Uout = 0.5*Uout ...
        + 0.5*(Ustar + h*K2);
end
end


function RHS = cs_conservative_transport_rhs( ...
    U,F,v,dx,dv)

[Nx,Nv] = size(U);

%% ============================================================
% x transport:
%
% d_x(v U)
%
% x is periodic
%% ============================================================

V = ones(Nx,1)*v(:).';

% Neighbor immediately to the right in x
UrightX = circshift(U,-1,1);

% Flux through x_{i+1/2}
FluxXplus = ...
    max(V,0).*U ...
    + min(V,0).*UrightX;

% Flux through x_{i-1/2}
FluxXminus = circshift(FluxXplus,1,1);

divX = (FluxXplus-FluxXminus)/dx;

%% ============================================================
% v transport:
%
% d_v(F U)
%% ============================================================

% Force at velocity interfaces v_{j+1/2}
Finterface = ...
    0.5*(F(:,1:end-1) + F(:,2:end));

% Values on the two sides of each interface
UleftV  = U(:,1:end-1);
UrightV = U(:,2:end);

% Upwind flux at internal velocity interfaces
FluxVinternal = ...
    max(Finterface,0).*UleftV ...
    + min(Finterface,0).*UrightV;

% There are Nv+1 velocity interfaces
FluxV = zeros(Nx,Nv+1);

FluxV(:,2:Nv) = FluxVinternal;

% Zero flux at the outer velocity boundaries
FluxV(:,1)    = 0;
FluxV(:,Nv+1) = 0;

divV = ...
    (FluxV(:,2:end)-FluxV(:,1:end-1))/dv;

%% Conservative transport equation
RHS = -divX-divV;
end
