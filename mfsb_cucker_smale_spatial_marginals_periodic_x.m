clear all;
close all;
clc;

%%

sigma2 = 1;
sigma  = sqrt(sigma2);
D      = sigma2/2;

xMin = -3; xMax = 3;
vMin = -6; vMax = 6;

Nx = 120;
Nv = 241;
Nt = 1000;

Lx = xMax - xMin;
x  = linspace(xMin,xMax,Nx+1)';
x(end) = [];
v  = linspace(vMin,vMax,Nv)';
dx = Lx/Nx;
dv = v(2)-v(1);

t  = linspace(0,1,Nt+1);
dt = 1/Nt;

ItersOut = 80;
ItersMid = 200;
tolMid   = 1e-5;
tolOut   = 1e-5;

maxSBInit = 150;
tolSBInit = 1e-5;

bcVarphi = 1e-12;
bcHat    = 1e-12;
bcMu     = 0;
bcRho    = 0;
posTol   = 0;
thr      = 1e-8;
theta    = 1;
clipH    = 1;

%% Endpoint specifications
phaseFloor = 1e-14;
distT = @(c) mod(x - c + 0.5*Lx,Lx) - 0.5*Lx;


x0Center = 0.0;
sx0      = 0.35;
rho_in_raw = exp(-0.5*(distT(x0Center)/sx0).^2);
rho_in = normalize1d_periodic(rho_in_raw,dx,bcRho);


xLeftT  = -0.8;
xRightT =  0.8;
sxT     =  0.25;
rho_fin_raw = 0.4*exp(-0.5*(distT(xLeftT )/sxT).^2) ...
            + 0.6*exp(-0.5*(distT(xRightT)/sxT).^2);
rho_fin = normalize1d_periodic(rho_fin_raw,dx,bcRho);


vLeftQ  = -1.5;
vRightQ =  1.5;
svQ     =  0.4;
MleftQ  = exp(-0.5*((v-vLeftQ )/svQ).^2);
MrightQ = exp(-0.5*((v-vRightQ)/svQ).^2);

muQ0_raw = rho_in_raw*(0.5*MleftQ + 0.5*MrightQ).';
muQ0_raw(:,2:Nv-1) = max(muQ0_raw(:,2:Nv-1),phaseFloor);
muQ0 = normalize2d_vboundary(muQ0_raw,dx,dv,bcMu,posTol,'muQ0');

rhoQ0 = sum(muQ0,2)*dv;
fprintf('prior initial spatial mismatch = %.3e\n', ...
    sum(abs(rhoQ0-rho_in))*dx);

%% Interaction Kernel
alphaCS = 0.45;
alignmentStrength = 3;

S = periodic_distance_matrix(x);
Wmat = alignmentStrength./(1 + S.^2).^alphaCS;

%% 
dtHalf = 0.5*dt;
AvHalfInt = diffusion_v_matrix(Nv,dtHalf,D,dv);
AvHalfIntDec = decomposition(AvHalfInt,'lu');
AvHalfAdjIntDec = decomposition(AvHalfInt.','lu');
[T0,~] = cs_affine_pullback(x,v,zeros(Nx,1),zeros(Nx,1),dt);

%% Initialization: Classical Sinkhorn
[mu_k,varphi_k,hatvarphi_k,eta1_k,errSBInit,lastSBInit,errEtaInit] = ...
    strang_SB_rho_endpoint( ...
    rho_in,rho_fin,muQ0,dv,dt,D,Nt, ...
    AvHalfIntDec,AvHalfAdjIntDec,T0, ...
    bcVarphi,bcHat,bcMu,posTol,thr,tolSBInit,maxSBInit);

muSB          = mu_k;
varphiSB      = varphi_k;
hatvarphiSB   = hatvarphi_k;
eta1SB        = eta1_k;

rhoSB0 = sum(muSB(:,:,1),2)*dv;
rhoSB1 = sum(muSB(:,:,end),2)*dv;

fprintf(['noninteracting spatial SB init | iters %d | err %.3e | ' ...
         'errEta %.3e | rho0 %.3e | rhoT %.3e\n'], ...
    lastSBInit,errSBInit(max(lastSBInit,1)), ...
    errEtaInit(max(lastSBInit,1)), ...
    hilbert_metric(rhoSB0,rho_in,thr,0), ...
    hilbert_metric(rhoSB1,rho_fin,thr,0));

%% Generalized Sinkhorn
errOut      = nan(ItersOut,1);
errMid      = nan(ItersMid,1);
errEtaMid   = nan(ItersMid,1);
massErr     = nan(ItersOut,1);
rhoEndErr0  = nan(ItersOut,1);
rhoEndErr1  = nan(ItersOut,1);

for k = 1:ItersOut
    rho_k = zeros(Nx,Nt+1);
    momentum_k = zeros(Nx,Nt+1);
    alignRate_k = zeros(Nx,Nt+1);
    weightedMomentum_k = zeros(Nx,Nt+1);

    for n = 1:Nt+1
        [rho_k(:,n),momentum_k(:,n),alignRate_k(:,n),weightedMomentum_k(:,n)] = ...
            cs_force_coefficients(mu_k(:,:,n),v,Wmat,dx,dv);
    end

    alignRateHalf_k = 0.5*(alignRate_k(:,1:Nt)+alignRate_k(:,2:Nt+1));
    weightedMomentumHalf_k = ...
        0.5*(weightedMomentum_k(:,1:Nt)+weightedMomentum_k(:,2:Nt+1));

    T_k = cell(Nt,1);
    clipMask_k = cell(Nt,1);
    for n = 1:Nt
        [T_k{n},clipMask_k{n}] = cs_affine_pullback( ...
            x,v,alignRateHalf_k(:,n),weightedMomentumHalf_k(:,n),dt);
    end

    maxClipMass = 0;
    for n = 1:Nt
        clippedMass_n = sum(mu_k(:,:,n+1).*clipMask_k{n},'all')*dx*dv;
        maxClipMass = max(maxClipMass,clippedMass_n);
    end
    fprintf('max clipped mass k=%d = %.3e\n',k,maxClipMass);

    varphi_j    = varphi_k;
    hatvarphi_j = hatvarphi_k;
    eta1_j      = eta1_k;

    errMid(:)    = nan;
    errEtaMid(:) = nan;

    for j = 1:ItersMid
     
        qSlope_j = zeros(Nx,Nt+1);
        qIntercept_j = zeros(Nx,Nt+1);
        ivQ = 3:Nv-2;

        for n = 1:Nt+1
            dvarphi_dv = centered_Dv_2D(varphi_j(:,:,n),dv);
            [qSlope_j(:,n),qIntercept_j(:,n)] = cs_q_coefficients( ...
                hatvarphi_j(:,:,n),dvarphi_dv,v,Wmat,dx,dv,ivQ);
        end
%Backward Eq.       
        varphi_next = bcVarphi*ones(Nx,Nv,Nt+1);
        varphi_next(:,:,end) = x_profile_to_phase(eta1_j,Nv,bcVarphi);

        for n = Nt+1:-1:2
            ell = n-1;
            qSlopeHalf = 0.5*(qSlope_j(:,ell)+qSlope_j(:,ell+1));
            qInterceptHalf = 0.5*(qIntercept_j(:,ell)+qIntercept_j(:,ell+1));

            Ghalf = cs_reaction_matrix(qSlopeHalf,qInterceptHalf, ...
                alignRateHalf_k(:,ell),v);

            varphi_next(:,:,n-1) = strang_adj_interacting( ...
                varphi_next(:,:,n),AvHalfAdjIntDec,T_k{ell},Ghalf, ...
                dt,D,dv,bcVarphi,posTol, ...
                sprintf('varphi adjoint spatial k=%d,j=%d,n=%d',k,j,n));
        end

        %hat_eta0(x)=rho_in / int muQ0*phi_0 dv.
        den0 = sum(muQ0.*varphi_next(:,:,1),2)*dv;
        hatEta0_next = spatial_ratio(rho_in,den0,posTol, ...
            sprintf('hatEta0 k=%d,j=%d',k,j));
%Forward Eq.
        hatvarphi_next = bcHat*ones(Nx,Nv,Nt+1);
        hat0 = muQ0.*hatEta0_next;
        hatvarphi_next(:,:,1) = impose_phase_boundary(hat0,bcHat);
        require_positive_phase(hatvarphi_next(:,:,1), ...
            sprintf('hatvarphi0 spatial k=%d,j=%d',k,j),posTol);

        % 
        for n = 1:Nt
            qSlopeHalf = 0.5*(qSlope_j(:,n)+qSlope_j(:,n+1));
            qInterceptHalf = 0.5*(qIntercept_j(:,n)+qIntercept_j(:,n+1));

            Ghalf = cs_reaction_matrix(qSlopeHalf,qInterceptHalf, ...
                alignRateHalf_k(:,n),v);

            hatvarphi_next(:,:,n+1) = strang_interacting( ...
                hatvarphi_next(:,:,n),AvHalfIntDec,T_k{n},Ghalf, ...
                dt,D,dv,bcHat,posTol, ...
                sprintf('hatvarphi forward spatial k=%d,j=%d,n=%d',k,j,n));
        end

        % eta1_next(x)=rho_fin / int hatphi_1 dv.
        den1 = sum(hatvarphi_next(:,:,end),2)*dv;
        eta1_next = spatial_ratio(rho_fin,den1,posTol, ...
            sprintf('eta1 k=%d,j=%d',k,j));

        varphi_next(:,:,end) = ...
            x_profile_to_phase(eta1_next,Nv,bcVarphi);

        
        err = hilbert_metric(eta1_next,eta1_j,thr,0);
        errEtaMid(j) = err;

        for n = 1:Nt+1
            err = max(err,hilbert_metric( ...
                varphi_next(:,:,n),varphi_j(:,:,n),thr,clipH));
            err = max(err,hilbert_metric( ...
                hatvarphi_next(:,:,n),hatvarphi_j(:,:,n),thr,clipH));
        end
        errMid(j) = err;

        varphi_j    = varphi_next;
        hatvarphi_j = hatvarphi_next;
        eta1_j      = eta1_next;

        if errMid(j) < tolMid
            break;
        end
    end

    varphi_kp1    = varphi_j;
    hatvarphi_kp1 = hatvarphi_j;
    eta1_kp1      = eta1_j;

    
    mu_kp1 = bcMu*ones(Nx,Nv,Nt+1);
    massTime = zeros(Nt+1,1);

    for n = 1:Nt+1
        cand = reconstruct_density( ...
            varphi_kp1(:,:,n),hatvarphi_kp1(:,:,n),bcMu,posTol, ...
            sprintf('candidate spatial k=%d,n=%d',k,n));

        massTime(n) = sum(cand,'all')*dx*dv;
        mu_kp1(:,:,n) = damp_density( ...
            cand,mu_k(:,:,n),theta,bcMu,posTol, ...
            sprintf('damped spatial mu k=%d,n=%d',k,n));
    end

    massErr(k) = max(abs(massTime-1));

    err = 0;
    for n = 1:Nt+1
        err = max(err,hilbert_metric( ...
            mu_kp1(:,:,n),mu_k(:,:,n),thr,clipH));
    end
    errOut(k) = err;

    rho0_kp1 = sum(mu_kp1(:,:,1),2)*dv;
    rho1_kp1 = sum(mu_kp1(:,:,end),2)*dv;

    rhoEndErr0(k) = hilbert_metric(rho0_kp1,rho_in,thr,0);
    rhoEndErr1(k) = hilbert_metric(rho1_kp1,rho_fin,thr,0);

    fprintf(['outer %3d | mid %3d | errOut %.3e | errMid %.3e | ' ...
             'errEta %.3e | massErr %.3e | rho0 %.3e | rhoT %.3e\n'], ...
        k,j,errOut(k),errMid(j),errEtaMid(j),massErr(k), ...
        rhoEndErr0(k),rhoEndErr1(k));

    mu_k         = mu_kp1;
    varphi_k     = varphi_kp1;
    hatvarphi_k  = hatvarphi_kp1;
    eta1_k       = eta1_kp1;

    if errOut(k) < tolOut
        break;
    end
end


mu            = mu_k;
varphi        = varphi_k;
hatvarphi     = hatvarphi_k;
eta1          = eta1_k;
lastOuter     = k;
errOutPlot    = errOut(1:lastOuter);
rhoEndErr0    = rhoEndErr0(1:lastOuter);
rhoEndErr1    = rhoEndErr1(1:lastOuter);

rho = squeeze(sum(mu,2))*dv;
mInitialInferred = sum(mu(:,:,1),1)'*dx;
mFinalInferred   = sum(mu(:,:,end),1)'*dx;

fprintf('\nFinal spatial endpoint diagnostics:\n');
fprintf('rho0 Hilbert error = %.3e\n', ...
    hilbert_metric(rho(:,1),rho_in,thr,0));
fprintf('rhoT Hilbert error = %.3e\n', ...
    hilbert_metric(rho(:,end),rho_fin,thr,0));
fprintf('initial full-law KL(mu0||muQ0) = %.8e\n', ...
    phase_relative_entropy(mu(:,:,1),muQ0,dx,dv,phaseFloor));

%% Save Cucker-Smale-interaction results

resultsFolder = fullfile(pwd,'saved_results');

if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder);
end

resultsFile = fullfile( ...
    resultsFolder, ...
    'Ex2_CuckerSmale_partial_results.mat');

%%
params = struct;

params.sigma2 = sigma2;
params.sigma  = sigma;
params.D      = D;

params.xMin = xMin;
params.xMax = xMax;
params.vMin = vMin;
params.vMax = vMax;

params.Nx = Nx;
params.Nv = Nv;
params.Nt = Nt;

params.dx = dx;
params.dv = dv;
params.dt = dt;

params.alphaCS = alphaCS;
params.alignmentStrength = alignmentStrength;

params.bcVarphi = bcVarphi;
params.bcHat    = bcHat;
params.bcMu     = bcMu;
params.bcRho    = bcRho;
params.posTol   = posTol;
params.thr      = thr;
params.theta    = theta;
params.clipH    = clipH;

params.ItersOut = ItersOut;
params.ItersMid = ItersMid;
params.tolMid   = tolMid;
params.tolOut   = tolOut;

params.maxSBInit = maxSBInit;
params.tolSBInit = tolSBInit;

%%
diagnostics = struct;

diagnostics.lastOuter = lastOuter;
diagnostics.errOut     = errOut(1:lastOuter);
diagnostics.massErr    = massErr(1:lastOuter);
diagnostics.rhoEndErr0 = rhoEndErr0(1:lastOuter);
diagnostics.rhoEndErr1 = rhoEndErr1(1:lastOuter);

diagnostics.lastSBInit = lastSBInit;
diagnostics.errSBInit  = errSBInit(1:lastSBInit);
diagnostics.errEtaInit = errEtaInit(1:lastSBInit);

%% Save variables needed for plotting
save(resultsFile, ...
    'mu', ...
    'varphi', ...
    'hatvarphi', ...
    'eta1', ...
    'muQ0', ...
    'rho_in', ...
    'rho_fin', ...
    'muSB', ...
    'varphiSB', ...
    'hatvarphiSB', ...
    'eta1SB', ...
    'mInitialInferred', ...
    'mFinalInferred', ...
    'x', ...
    'v', ...
    't', ...
    'Wmat', ...
    'params', ...
    'diagnostics', ...
    '-v7.3');


%% Local functions
function [mu_sb,varphi_sb,hatvarphi_sb,eta1, ...
          errSB,lastIter,errEta] = ...
    strang_SB_rho_endpoint( ...
    rho_in,rho_fin,muQ0,dv,dt,D,Nt, ...
    AvHalfIntDec,AvHalfAdjIntDec,T0, ...
    bcValVarphi,bcValHat,bcMu,posTol,thr,tolSB,maxIter)

    [Nx,Nv] = size(muQ0);

    iv = 2:Nv-1;

    eta1 = ones(Nx,1);

    errSB  = nan(maxIter,1);
    errEta = nan(maxIter,1);

    lastIter = maxIter;

    varphi_old    = bcValVarphi*ones(Nx,Nv,Nt+1);
    hatvarphi_old = bcValHat*ones(Nx,Nv,Nt+1);

    for iter = 1:maxIter

        varphi_next = bcValVarphi*ones(Nx,Nv,Nt+1);

        varphi_next(:,:,end) = ...
            x_profile_to_phase(eta1,Nv,bcValVarphi);

        for tt = Nt+1:-1:2

            varphi_next(:,:,tt-1) = ...
                strang_adj_interacting( ...
                varphi_next(:,:,tt), ...
                AvHalfAdjIntDec, ...
                T0, ...
                zeros(Nx,1), ...
                dt,D,dv, ...
                bcValVarphi,posTol, ...
                'SB spatial adjoint');

        end

 
        den0 = sum( ...
            muQ0(:,iv).*varphi_next(:,iv,1), ...
            2)*dv;

        hatEta0 = spatial_ratio( ...
            rho_in,den0,posTol, ...
            'SB spatial hatEta0');

        hatvarphi_next = ...
            bcValHat*ones(Nx,Nv,Nt+1);

        hat0 = muQ0.*hatEta0;

        hatvarphi_next(:,:,1) = ...
            impose_phase_boundary(hat0,bcValHat);

        require_positive_phase( ...
            hatvarphi_next(:,:,1), ...
            'SB spatial hatvarphi0', ...
            posTol);


        for tt = 1:Nt

            hatvarphi_next(:,:,tt+1) = ...
                strang_interacting( ...
                hatvarphi_next(:,:,tt), ...
                AvHalfIntDec, ...
                T0, ...
                zeros(Nx,1), ...
                dt,D,dv, ...
                bcValHat,posTol, ...
                'SB spatial forward');

        end

   
        den1 = sum( ...
            hatvarphi_next(:,iv,end), ...
            2)*dv;

        eta1_next = spatial_ratio( ...
            rho_fin,den1,posTol, ...
            'SB spatial eta1');

       
        varphi_next(:,:,end) = ...
            x_profile_to_phase( ...
            eta1_next,Nv,bcValVarphi);


        errEta(iter) = ...
            hilbert_metric(eta1_next,eta1,thr,0);

        if iter > 1

            err = errEta(iter);

            for tt = 1:Nt+1

                err = max(err, ...
                    hilbert_metric( ...
                    varphi_next(:,:,tt), ...
                    varphi_old(:,:,tt), ...
                    thr,2));

                err = max(err, ...
                    hilbert_metric( ...
                    hatvarphi_next(:,:,tt), ...
                    hatvarphi_old(:,:,tt), ...
                    thr,2));

            end

        else

            err = Inf;

        end

        errSB(iter) = err;

        eta1          = eta1_next;
        varphi_old    = varphi_next;
        hatvarphi_old = hatvarphi_next;

        if errSB(iter) < tolSB

            lastIter = iter;
            break;

        end

    end

    varphi_sb    = varphi_old;
    hatvarphi_sb = hatvarphi_old;

    mu_sb = bcMu*ones(Nx,Nv,Nt+1);

    for tt = 1:Nt+1

        mu_sb(:,:,tt) = ...
            reconstruct_density( ...
            varphi_sb(:,:,tt), ...
            hatvarphi_sb(:,:,tt), ...
            bcMu,posTol, ...
            sprintf('SB spatial mu tt=%d',tt));

    end

end

function U = x_profile_to_phase(a,Nv,bcVal)
    U = a(:)*ones(1,Nv);
    U = impose_phase_boundary(U,bcVal);
end

function r = spatial_ratio(num,den,posTol,name)
    num = num(:);
    den = den(:);

    if any(~isfinite(num)) || any(~isfinite(den))
        error('%s has nonfinite entries.',name);
    end
    if any(num <= posTol) || any(den <= posTol)
        error('%s has nonpositive numerator or denominator.',name);
    end

    r = num./den;
    if any(~isfinite(r)) || any(r <= posTol)
        error('%s produced a nonpositive ratio.',name);
    end
end

function rho = normalize1d_periodic(rho,dx,bcVal)
    rho = max(rho,bcVal);
    mass = sum(rho)*dx;
    if ~isfinite(mass) || mass <= 0
        error('Bad periodic one-dimensional normalization.');
    end
    rho = rho/mass;
end

function val = phase_relative_entropy(mu0,muQ0,dx,dv,floorVal)
    A = max(mu0,floorVal);
    B = max(muQ0,floorVal);
    val = sum(A.*log(A./B),'all')*dx*dv;
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
    U(:,1) = bcVal;
    U(:,end) = bcVal;
end

function require_positive_phase(U,name,posTol)
    if any(~isfinite(U(:)))
        error('%s has nonfinite entries.',name);
    end

   
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


function d = hilbert_metric(U,V,thr,clipH)
    if nargin < 4
        clipH = 0;
    end

    if isvector(U)
        U = U(:);
        V = V(:);
    else
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

