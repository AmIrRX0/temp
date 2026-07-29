`gatherlib` is a small package of Triton kernels that read and write whole rows of a two dimensional table on the GPU. Three operations are exported, and `gatherlib.run_op(name, ...)` reaches all three of them by name:

- `gather_rows(table, row_ids)` returns a packed `(len(row_ids), table.shape[1])` copy of the named rows.
- `row_sums(table, row_ids)` returns a float32 vector holding the sum of each named row.
- `scatter_rows_(table, row_ids, values)` writes whole rows back into the table in place.

A table here is any two dimensional CUDA tensor. It does not have to own its storage, so it may be a view of some other tensor, and it may be as large as the device is able to hold. `row_ids` is a one dimensional integer tensor.

Contract 1: `gather_rows(table, row_ids)` must return a tensor whose row *i* holds exactly the elements that `table[row_ids[i]]` holds. That has to be true for every table size the device can fit and for every valid memory layout the table is allowed to have. Two tables which hold equal data must therefore gather to equal results, even if the two of them are laid out differently in memory.

Contract 2: none of the three operations may read or write memory outside the region covered by the table it was handed.

Right now Contract 1 does not hold for every input. Where it breaks, the returned tensor still has the right shape, the right dtype and the right device, nothing is raised and nothing is warned, but some of the values it returns are not the values that were asked for.

Nothing else may regress. `row_sums` and `scatter_rows_` keep the behaviour they have, and so does every input that `gather_rows` already handles. The output of `gather_rows` keeps its shape, dtype, device and contiguity contract. A row id outside the table keeps raising `IndexError`. The `block_size` knob on `GatherConfig` keeps having no effect at all on the values that come back. The registry keeps exposing the same three names.
