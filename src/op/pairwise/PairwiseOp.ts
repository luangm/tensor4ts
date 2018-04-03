import Operation from "../Operation";
import Tensor from "../../Tensor";

export default abstract class PairwiseOp extends Operation {

  private readonly _left: Tensor;
  private readonly _result: Tensor;
  private readonly _right: Tensor;

  get left() {
    return this._left;
  }

  get result() {
    return this._result;
  }

  get right() {
    return this._right;
  }

  protected constructor(left: Tensor, right: Tensor, result: Tensor) {
    super([left, right], [result]);
    this._left = left;
    this._right = right;
    this._result = result;
  }

  abstract body(a: number, b: number): number;

  exec(dim?: number | number[]): void {
    // nothing
  }

}