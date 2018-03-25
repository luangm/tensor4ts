import Operation from "../Operation"
import Tensor from "../../Tensor";

export default class LinspaceOp extends Operation {

  private _num: number;
  private _start: number;
  private _step: number;
  private _stop: number;

  constructor(input: Tensor, other: Tensor, result: Tensor, start: number = 0, stop: number = 1, num: number = 1) {
    super(input, other, result);
    this._start = start;
    this._stop = stop;
    this._num = num;
    this._step = num === 1 ? 0 : (stop - start) / (num - 1);
  }

  get isSpecial() {
    return true;
  }

  exec() {
    let result = this.result.data;
    let val = this._start;
    for (let i = 0; i < result.length; i++) {
      result[i] = val;
      val += this._step;
    }
  }

}