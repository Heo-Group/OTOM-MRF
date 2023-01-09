clear;clc;

%%
max_label=[100,   0.17,  100e-6,  3.0];
min_label=[  5,   0.02,    1e-6,  0.2];

load('Phantom_Kex_rnn_mtc_LOAS40_with_B0_1p2_rB1_0p5_corrected.mat');
result_nn=result_nn.*(max_label-min_label)+min_label;
result_BE=reshape(result_nn,[256 256 4]);
Rm_BE=squeeze(result_BE(:,:,1));

load('Phantom_Mom_rnn_mtc_LOAS40_with_B0_1p2_rB1_0p5_corrected.mat');
result_nn=result_nn.*(max_label-min_label)+min_label;
result_BE=reshape(result_nn,[256 256 4]);
Mm_BE=squeeze(result_BE(:,:,2));

load('Phantom_T2m_rnn_mtc_LOAS40_with_B0_1p2_rB1_0p5_corrected.mat');
result_nn=result_nn.*(max_label-min_label)+min_label;
result_BE=reshape(result_nn,[256 256 4]);
T2m_BE=squeeze(result_BE(:,:,3));

load('Phantom_T1w_rnn_mtc_LOAS40_with_B0_1p2_rB1_0p5_corrected.mat');
result_nn=result_nn.*(max_label-min_label)+min_label;
result_BE=reshape(result_nn,[256 256 4]);
T1w_BE=squeeze(result_BE(:,:,4));

load('mask_phantom.mat');
Rm_BE(~mask_phantom)=0;
Mm_BE(~mask_phantom)=0;
T2m_BE(~mask_phantom)=0;
T1w_BE(~mask_phantom)=0;
%%
figure2   = figure;
ax1   = axes( 'Parent', figure2 );
ax2   = axes( 'Parent', figure2 );
set( ax1, 'Visible', 'off' )
set( ax2, 'Visible', 'off' )
h1   = imshow( Rm_BE, [5 80], 'Parent', ax1 );
h2   = imshow( mask_phantom, [0 1], 'Parent', ax2 );
set(h1, 'AlphaData',mask_phantom );
set(h2, 'AlphaData',~mask_phantom );
colormap(ax1, parula);
colormap(ax2,gray);

%%  Mm
figure2   = figure;
ax1   = axes( 'Parent', figure2 );
ax2   = axes( 'Parent', figure2 );
set( ax1, 'Visible', 'off' )
set( ax2, 'Visible', 'off' )
h1   = imshow( Mm_BE, [0.02 0.17], 'Parent', ax1 );
h2   = imshow( mask_phantom, [0 1], 'Parent', ax2 );
set(h1, 'AlphaData',mask_phantom );
set(h2, 'AlphaData',~mask_phantom );
colormap(ax1, parula);
colormap(ax2,gray);

%%  T2m
figure2   = figure;
ax1   = axes( 'Parent', figure2 );
ax2   = axes( 'Parent', figure2 );
set( ax1, 'Visible', 'off' )
set( ax2, 'Visible', 'off' )
h1   = imshow( T2m_BE, [1e-6 1e-4], 'Parent', ax1 );
h2   = imshow( mask_phantom, [0 1], 'Parent', ax2 );
set(h1, 'AlphaData',mask_phantom );
set(h2, 'AlphaData',~mask_phantom );
colormap(ax1, parula);
colormap(ax2,gray);

%%  T1w
figure2   = figure;
ax1   = axes( 'Parent', figure2 );
ax2   = axes( 'Parent', figure2 );
set( ax1, 'Visible', 'off' )
set( ax2, 'Visible', 'off' )
h1   = imshow( T1w_BE, [0.5 3.0], 'Parent', ax1 );
h2   = imshow( mask_phantom, [0 2.7], 'Parent', ax2 );
set(h1, 'AlphaData',mask_phantom );
set(h2, 'AlphaData',~mask_phantom );
colormap(ax1, parula);
colormap(ax2,gray);
