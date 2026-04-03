"""Reference paper results and helpers to compare a run against them."""

import argparse
import json


PAPER_RESULTS = {
    'fashion200k_retrieval': {
        'TIRG': {
            'recall_top1_correct_composition': 0.141,
            'recall_top10_correct_composition': 0.425,
            'recall_top50_correct_composition': 0.638,
        },
    },
    'mitstates_retrieval': {
        'TIRG': {
            'recall_top1_correct_composition': 0.122,
            'recall_top5_correct_composition': 0.319,
            'recall_top10_correct_composition': 0.431,
        },
    },
    'mitstates_classification': {
        'TIRG': {
            'accuracy': 0.152,
        },
    },
    'css3d_retrieval': {
        '3d': {
            'TIRG': {
                'recall_top1_correct_composition': 0.737,
            },
            'TIRGLastConv': {
                'recall_top1_correct_composition': 0.737,
            },
            'Concat': {
                'recall_top1_correct_composition': 0.606,
            },
        },
        '2d': {
            'TIRG': {
                'recall_top1_correct_composition': 0.466,
            },
            'Concat': {
                'recall_top1_correct_composition': 0.273,
            },
        },
    },
    'ablation': {
        'fashion200k': {
            'Our Full Model': 0.141,
            '- gated feature only': 0.139,
            '- residue feature only': 0.121,
            '- mod. at last fc': 0.141,
            '- mod. at last conv': 0.124,
            'DML loss, K = 2': 0.095,
            'DML loss, K = B': 0.141,
        },
        'mitstates': {
            'Our Full Model': 0.122,
            '- gated feature only': 0.071,
            '- residue feature only': 0.119,
            '- mod. at last fc': 0.122,
            '- mod. at last conv': 0.103,
            'DML loss, K = 2': 0.122,
            'DML loss, K = B': 0.109,
        },
        'css3d': {
            'Our Full Model': 0.737,
            '- gated feature only': 0.065,
            '- residue feature only': 0.606,
            '- mod. at last fc': 0.712,
            '- mod. at last conv': 0.737,
            'DML loss, K = 2': 0.737,
            'DML loss, K = B': 0.698,
        },
    },
}

REIMPLEMENTATION_RESULTS = {
    'css3d': {
        'TIRG': {
            'recall_top1_correct_composition': 0.760,
        },
    },
    'fashion200k': {
        'TIRG': {
            'recall_top1_correct_composition': 0.161,
        },
    },
    'mitstates': {
        'TIRG': {
            'recall_top1_correct_composition': 0.132,
        },
    },
}


def get_reference(args):
  """Returns the relevant reference metric dictionary."""
  if args.source == 'repo':
    return REIMPLEMENTATION_RESULTS[args.dataset][args.row]
  if args.experiment == 'retrieval':
    if args.dataset == 'css3d':
      return PAPER_RESULTS['css3d_retrieval'][args.query_mode][args.row]
    return PAPER_RESULTS[f'{args.dataset}_retrieval'][args.row]
  if args.experiment == 'classification':
    return PAPER_RESULTS[f'{args.dataset}_classification'][args.row]
  raise ValueError('Unsupported experiment %s' % args.experiment)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--metrics_json', type=str, default='')
  parser.add_argument(
      '--dataset',
      type=str,
      required=True,
      choices=['css3d', 'fashion200k', 'mitstates'])
  parser.add_argument(
      '--experiment',
      type=str,
      default='retrieval',
      choices=['retrieval', 'classification'])
  parser.add_argument('--row', type=str, default='TIRG')
  parser.add_argument(
      '--query_mode', type=str, default='3d', choices=['2d', '3d'])
  parser.add_argument('--split', type=str, default='test')
  parser.add_argument('--source', type=str, default='paper', choices=['paper', 'repo'])
  args = parser.parse_args()

  reference = get_reference(args)
  print('Reference (%s):' % args.source)
  for metric_name, metric_value in reference.items():
    print('  %s: %.4f' % (metric_name, metric_value))

  if not args.metrics_json:
    return

  with open(args.metrics_json) as f:
    metrics = json.load(f)
  actual = metrics[args.split]

  print('')
  print('Comparison for %s:' % args.metrics_json)
  for metric_name, expected_value in reference.items():
    actual_value = actual.get(metric_name)
    if actual_value is None:
      print('  %s: missing from run output' % metric_name)
      continue
    diff = actual_value - expected_value
    print('  %s: actual=%.4f expected=%.4f diff=%+.4f' %
          (metric_name, actual_value, expected_value, diff))


if __name__ == '__main__':
  main()
