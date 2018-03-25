import Operation from "../Operation";
import Tensor from "../../Tensor";

export default abstract class IndexOp extends Operation {

  private _dim: number;

  constructor(input: Tensor, other: Tensor, result: Tensor, dim: number) {
    super(input, other, result);
    this._dim = dim;
  }

  get dim() {
    return this._dim;
  }

  exec(dim?: number): void {
    // nothing
  }

  abstract update(accum: number, a: number, accumIdx: number, idx: number): [number, number];
}
