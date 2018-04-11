import Tensor from "../../Tensor";
import Operation from "../Operation";

export default abstract class NaryOp extends Operation {

  private readonly _list: Tensor[];
  private readonly _result: Tensor;

  get list() {
    return this._list;
  }

  get result() {
    return this._result;
  }

  protected constructor(list: Tensor[], result: Tensor) {
    super(list, [result]);
    this._list = list;
    this._result = result;
  }

  abstract body(values: number[]): number;

  exec(dim?: number | number[]): void {
    // nothing
  }
}