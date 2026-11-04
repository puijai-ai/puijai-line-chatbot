# -*- coding: utf-8 -*-
"""
Generate publication-quality figures for Puijai Senior Project Thesis & Progress Report.
Saves high-res PNGs to docs/figures/ in both repository folders.
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Tahoma', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIRS = [
    r'c:\Users\USER\Downloads\แชทบอท\docs\figures',
    r'c:\ML\ML\New folder\Puijai\docs\figures'
]

def save_fig(fig, filename):
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

# ==============================================================================
# 1. Confusion Matrix Heatmap (Table 4.3, N=600)
# ==============================================================================
def plot_confusion_matrix():
    cm = np.array([
        [191,   7,   2],
        [  4, 184,  12],
        [  1,   6, 193]
    ])
    classes = ['Normal\n(Level 0)', 'Early Risk\n(Level 1)', 'Crisis\n(Level 2)']
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=classes, yticklabels=classes,
                annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
    
    ax.set_title('Confusion Matrix - WangchanBERTa (K=3, L=128)\nTest Set (N=600, Accuracy=94.67%)', fontsize=13, weight='bold', pad=15)
    ax.set_xlabel('Predicted Risk Class', fontsize=11, weight='bold')
    ax.set_ylabel('Actual Ground-Truth Risk Class', fontsize=11, weight='bold')
    
    # Add annotation for Crisis row
    plt.tight_layout()
    save_fig(fig, 'fig_4_1_confusion_matrix.png')
    plt.close()

# ==============================================================================
# 2. 3x3 Factorial Sliding Window Optimization (Table 4.2, N=300 Validation Set)
# ==============================================================================
def plot_sliding_window_experiment():
    experiments = [
        'Exp-1\n(K=1,L=64)', 'Exp-2\n(K=1,L=128)', 'Exp-3\n(K=1,L=256)',
        'Exp-4\n(K=3,L=64)', 'Exp-5 (Optimal)\n(K=3,L=128)', 'Exp-6\n(K=3,L=256)',
        'Exp-7\n(K=5,L=64)', 'Exp-8\n(K=5,L=128)', 'Exp-9\n(K=5,L=256)'
    ]
    recall = [78.50, 80.00, 80.50, 89.00, 96.50, 96.00, 86.50, 92.50, 90.00]
    f1 =     [82.43, 83.77, 84.10, 90.13, 94.84, 94.35, 88.26, 91.13, 88.67]
    latency = [38.4,  45.2,  68.9,  44.1,  56.8,  89.4,  48.5,  64.3,  98.7]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    x = np.arange(len(experiments))
    w = 0.35

    # Bar chart for Recall & F1
    rects1 = ax1.bar(x - w/2, recall, w, label='Crisis Recall (%)', color='#2b5c8f')
    rects2 = ax1.bar(x + w/2, f1, w, label='Crisis F1-Score (%)', color='#4ba375')
    
    # Highlight optimal Exp-5
    rects1[4].set_color('#d9534f')
    rects2[4].set_color('#f0ad4e')
    
    ax1.axhline(90.0, color='gray', linestyle='--', linewidth=1, label='Target Goal (90%)')
    ax1.set_ylabel('Score (%)', fontsize=11, weight='bold')
    ax1.set_title('3x3 Factorial Design: Effect of Sliding Window (K Turns, L Tokens)\non Crisis Point Detection (Validation Set N=300)', fontsize=12, weight='bold')
    ax1.set_ylim(70, 102)
    ax1.legend(loc='lower left', frameon=True)
    
    # Annotate Exp-5
    ax1.annotate('Optimal Point (Exp-5)\nRecall: 96.50%, F1: 94.84%',
                 xy=(4, 96.50), xytext=(4, 100),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 ha='center', fontsize=9, weight='bold', color='#d9534f')

    # Line chart for Latency
    ax2.plot(x, latency, marker='o', color='#c9302c', linewidth=2, label='Inference Latency (ms)')
    ax2.plot(4, 56.8, marker='*', markersize=14, color='#d9534f')
    ax2.set_ylabel('Latency (ms)', fontsize=11, weight='bold')
    ax2.set_xlabel('Sliding Window Experimental Configurations', fontsize=11, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(experiments, fontsize=9)
    ax2.set_ylim(20, 120)
    ax2.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    save_fig(fig, 'fig_4_2_sliding_window_optimization.png')
    plt.close()

# ==============================================================================
# 3. Model vs Baseline Comparison (Table 4.1)
# ==============================================================================
def plot_baseline_comparison():
    models = [
        'Single-Turn\nKeyword Matching',
        'Multi-Turn (K=3)\nContext Keyword',
        'WangchanBERTa\nSingle-Turn (K=1)',
        'WangchanBERTa\nSliding Window (K=3,L=128)'
    ]
    recall = [57.50, 68.00, 80.00, 96.50]
    precision = [71.88, 77.27, 87.91, 93.24]
    f1 = [63.89, 72.34, 83.77, 94.84]
    false_negatives = [85, 64, 40, 7]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(models))
    w = 0.25

    ax1.bar(x - w, precision, w, label='Crisis Precision (%)', color='#5bc0de')
    ax1.bar(x, recall, w, label='Crisis Recall (%)', color='#0275d8')
    ax1.bar(x + w, f1, w, label='Crisis F1-Score (%)', color='#5cb85c')

    ax1.set_ylabel('Performance (%)', fontsize=11, weight='bold')
    ax1.set_ylim(40, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, weight='bold')
    ax1.set_title('Model Performance Comparison vs Baseline (Test Set N=600)\nDemonstrating Contextual Sliding Window Advantage', fontsize=12, weight='bold')
    ax1.legend(loc='upper left')

    # Secondary axis for False Negatives
    ax2 = ax1.twinx()
    ax2.plot(x, false_negatives, color='#d9534f', marker='s', linewidth=2.5, label='False Negatives (Cases missed)')
    ax2.set_ylabel('Crisis False Negatives (Missed Cases)', color='#d9534f', fontsize=11, weight='bold')
    ax2.tick_params(axis='y', labelcolor='#d9534f')
    ax2.set_ylim(0, 100)
    for i, val in enumerate(false_negatives):
        ax2.annotate(f'{val} cases', xy=(i, val), xytext=(i, val + 4),
                     ha='center', fontsize=9, weight='bold', color='#d9534f')

    plt.tight_layout()
    save_fig(fig, 'fig_4_3_model_vs_baseline.png')
    plt.close()

# ==============================================================================
# 4. System Usability Scale (SUS) Item Breakdown (Table 4.5, N=30)
# ==============================================================================
def plot_sus_scores():
    items = [
        'Q1: I would like to use Puijai frequently',
        'Q2: I found the system unnecessarily complex*',
        'Q3: I thought the system was easy to use',
        'Q4: I think I would need technical support*',
        'Q5: The functions were well integrated',
        'Q6: I thought there was too much inconsistency*',
        'Q7: Most people would learn to use it quickly',
        'Q8: I found the system very cumbersome to use*',
        'Q9: I felt very confident using the system',
        'Q10: I needed to learn a lot before getting going*'
    ]
    raw_scores = [4.53, 1.47, 4.60, 1.33, 4.47, 1.57, 4.67, 1.40, 4.40, 1.60]
    adj_scores = [3.53, 3.53, 3.60, 3.67, 3.47, 3.43, 3.67, 3.60, 3.40, 3.40]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(items))

    colors = ['#28a745' if (i % 2 == 0) else '#17a2b8' for i in range(len(items))]
    bars = ax.barh(y, adj_scores, color=colors, height=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels(items, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel('Adjusted Score Contribution (Scale 0 to 4)', fontsize=11, weight='bold')
    ax.set_xlim(0, 4.8)
    ax.set_title('System Usability Scale (SUS) Item Breakdown (N=30 Participants)\nAdjusted Sum = 34.70 -> Final SUS Score = 34.70 x 2.5 = 86.75 (Grade A, Excellent)', fontsize=12, weight='bold')

    for bar, raw, adj in zip(bars, raw_scores, adj_scores):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
                f'Raw Mean: {raw:.2f} | Adj: {adj:.2f}',
                va='center', fontsize=9, weight='bold')

    plt.tight_layout()
    save_fig(fig, 'fig_4_4_sus_item_breakdown.png')
    plt.close()

# ==============================================================================
# 5. Architecture Schematic Diagram (Figure 3.1)
# ==============================================================================
def plot_system_architecture():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis('off')

    # Draw boxes
    boxes = [
        ('LINE Client\n(User UI)', 0.08, 0.5, 0.14, 0.35, '#e1f5fe', '#0288d1'),
        ('LINE Platform\n(Messaging API)', 0.28, 0.5, 0.14, 0.35, '#e8f5e9', '#388e3c'),
        ('FastAPI Webhook\nServer (Port 8000)', 0.48, 0.5, 0.16, 0.5, '#fff3e0', '#f57c00'),
        ('Context Sliding\nWindow (K=3, L=128)', 0.70, 0.7, 0.16, 0.25, '#f3e5f5', '#7b1fa2'),
        ('WangchanBERTa\nCrisis Classifier', 0.70, 0.3, 0.16, 0.25, '#ede7f6', '#512da8'),
        ('Emergency 1323\nFlex Message Action', 0.90, 0.5, 0.14, 0.35, '#ffebee', '#d32f2f'),
    ]

    for label, x, y, w, h, fill, border in boxes:
        rect = plt.Rectangle((x - w/2, y - h/2), w, h, facecolor=fill, edgecolor=border, linewidth=2, transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9.5, weight='bold', transform=ax.transAxes, zorder=3)

    # Draw connecting arrows
    arrows = [
        ((0.15, 0.5), (0.21, 0.5), 'Chat Msg'),
        ((0.35, 0.5), (0.40, 0.5), 'Webhook POST\n(HMAC Verified)'),
        ((0.56, 0.62), (0.62, 0.7), 'Turns'),
        ((0.78, 0.7), (0.78, 0.45), 'L=128 Tokens'),
        ((0.70, 0.17), (0.56, 0.35), 'Pred: Crisis'),
        ((0.56, 0.5), (0.83, 0.5), 'Trigger 1323 Flex')
    ]

    for (x1, y1), (x2, y2), text in arrows:
        ax.annotate(text, xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='#333333', lw=1.5),
                    ha='center', va='center', fontsize=8, weight='bold',
                    xycoords='axes fraction', textcoords='axes fraction', zorder=4)

    ax.set_title('Puijai Chatbot: End-to-End Crisis Point Detection Architecture (System Flow)',
                 fontsize=13, weight='bold', pad=20)

    plt.tight_layout()
    save_fig(fig, 'fig_3_1_system_architecture.png')
    plt.close()

if __name__ == '__main__':
    print("Generating report figures...")
    plot_confusion_matrix()
    plot_sliding_window_experiment()
    plot_baseline_comparison()
    plot_sus_scores()
    plot_system_architecture()
    print("All 5 publication figures generated successfully!")
