from unittest.mock import MagicMock

from masked_stellar_autoencoder.models.model import TabResnetWrapper


def test_resume_runs_until_total_epoch_target():
    wrapper = TabResnetWrapper.__new__(TabResnetWrapper)
    wrapper._setup_pretrain_optimizer = MagicMock(return_value=(None, None))
    wrapper._configure_pretrain_logging = MagicMock()
    wrapper._load_pretrain_resume = MagicMock(return_value=(0.0, 0.0, 3))
    wrapper._run_pretrain_epoch = MagicMock(return_value=(1.0, 1.0))
    wrapper._chain_finetune_after_pretrain = MagicMock()

    wrapper.pretrain_hdf(
        train_keys=["train"], num_epochs=5, mini_batch=1, pretrained="checkpoint.pth"
    )

    calls = wrapper._run_pretrain_epoch.call_args_list
    assert [call.args[0] for call in calls] == [3, 4]
    assert all(call.args[1] == 5 for call in calls)
