"""PolaRiS baseline policy configs."""

from typing import TypeAlias

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.droid_policy as droid_policy
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType


def get_polaris_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import AssetsConfig
    from openpi.training.config import SimpleDataConfig
    from openpi.training.config import TrainConfig

    return [
        #
        # PolaRiS DROID jointpos policies 
        #
        TrainConfig(
            name="pi05_droid_jointpos",
            model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],       # type: ignore
                    outputs=[
                        _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                        droid_policy.DroidOutputs(),
                    ],
                ),
            ),
        ),

        TrainConfig(
            name="pi0_fast_droid_jointpos",
            model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(    # type: ignore
                    inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                    outputs=[
                        _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                        droid_policy.DroidOutputs(),
                    ],
                ),
            ),
        ),
        TrainConfig(
            name="pi0_droid_jointpos",
            model=pi0_config.Pi0Config(
                # action_dim is left as 32 default...
                action_horizon=10,
                max_token_len=100,
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(  
                    inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                    outputs=[
                        _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                        droid_policy.DroidOutputs(),
                    ],
                ),
            ),
        ),

        TrainConfig(
            name="paligemma_binning_droid_jointpos_fullfinetune",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8, 
                action_horizon=15, 
                max_token_len=600,
                fast_model_tokenizer=_tokenizer.BinningTokenizer,
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                    outputs=[
                        _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                        droid_policy.DroidOutputs(),
                    ],
                ),
            ),
        ),
    ]
