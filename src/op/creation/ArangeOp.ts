import Operation from "../Operation"
import Tensor from "../../Tensor";

export default class ArangeOp extends Operation {
  private _start: number;
  private _step: number;
  private _stop: number;

  constructor(input: Tensor, other: Tensor, result: Tensor, stop: number, start: number = 0, step: number = 1) {
    super(input, other, result);
    this._start = start;
    this._stop = stop;
    this._step = step;
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