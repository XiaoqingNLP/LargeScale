"""Evaluation"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args
from megatron.initialize import initialize_megatron

from evaluate import main


def get_evaluation_args(parser):
    """Provide extra arguments required for evaluation."""
    group = parser.add_argument_group(title='evaluation')
    group.add_argument('--eval-data-path', type=str, required=True)
    group.add_argument('--task', nargs='+', default=None, help='Task to evaluate')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')

    return parser


if __name__ == '__main__':

    initialize_megatron(extra_args_provider=get_evaluation_args)

    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for downstream tasks.")
        exit()

    main()
