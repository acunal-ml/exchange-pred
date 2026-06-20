import torch

from ml_pipeline.lstm_model import AttentionLSTM


def test_forward_output_shapes():
    batch, seq_len, n_features, num_classes = 4, 12, 9, 3
    model = AttentionLSTM(n_features=n_features, hidden_size=16, num_layers=2, dropout=0.1, num_classes=num_classes)
    x = torch.randn(batch, seq_len, n_features)

    logits, attn_weights = model(x)

    assert logits.shape == (batch, num_classes)
    assert attn_weights.shape == (batch, seq_len)


def test_attention_weights_sum_to_one():
    batch, seq_len, n_features = 3, 8, 5
    model = AttentionLSTM(n_features=n_features, hidden_size=8, num_layers=1, dropout=0.0)
    x = torch.randn(batch, seq_len, n_features)

    _, attn_weights = model(x)
    sums = attn_weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(batch), atol=1e-5)


def test_single_layer_lstm_ignores_dropout_param_without_error():
    # nn.LSTM raises a warning (not error) if dropout>0 with num_layers=1;
    # the model must guard against passing dropout through in that case.
    model = AttentionLSTM(n_features=4, hidden_size=8, num_layers=1, dropout=0.5)
    assert model.lstm.dropout == 0.0
