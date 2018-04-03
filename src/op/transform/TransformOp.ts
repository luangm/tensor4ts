import Tensor from "../../Tensor";
import Operation from "../Operation";

export default abstract class TransformOp extends Operation {

  private readonly _base: Tensor;
  private readonly _result: Tensor;

  get base() {
    return this._base;
  }

  get result() {
    return this._result;
  }

  protected constructor(base: Tensor, result: Tensor) {
    super([base], [result]);
    this._base = base;
    this._result = result;
  }

  abstract body(a: number): number;

  exec(dim?: number | number[]): void {
    // nothing
  }

}