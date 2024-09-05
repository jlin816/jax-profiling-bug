def main():
    import jax
    import numpy as np
    import granular
    from big_vision.datasets.interleaved.interleaved import make_interleaved_mixture
    from big_vision import utils as u
    P = jax.sharding.PartitionSpec

    jax.distributed.initialize()

    mesh = jax.sharding.Mesh(jax.devices(), "d")
    d = jax.sharding.NamedSharding(mesh, P("d"))
    n = jax.sharding.NamedSharding(mesh, P())

    # Define dummy update function.
    def fn(x, y):
        y = y["image"][:, :, 0, 0, 0].repeat(256 // 16, 1)
        res = x @ y.repeat(len(x) // len(y), 0)
        loss = res.sum()
        return res, loss

    fn = jax.jit(fn, in_shardings=d, out_shardings=(d, n))
    x = jax.device_put(np.ones((256, 256)), d)

    dataset = ... # Expensive dataset that loads images from GCS
    loader = ... # Multiprocessing dataloader with large batch size
    it = iter(loader)

    start_step = 5
    end_step = 7  # Setting this to 25 results in no TPU ops on profile

    for i, y in zip(range(30), it):
      if i == start_step:
        jax.profiler.start_trace('profiles')

      if i > 0:
        prev_loss = loss
      with jax.profiler.StepTraceAnnotation('train', step_num=i):
        res, loss = fn(x, y)
      # Runahead max one batch.
      if i > 0:
        jax.block_until_ready(prev_loss)

      if i == 7:
        jax.profiler.stop_trace()


if __name__ == '__main__':
    main()