import Operation from "../Operation";
import Tensor from "../../Tensor";

export default class ArangeOp extends Operation {

  private readonly _result: Tensor;
  private readonly _start: number;
  private readonly _step: number;
  private readonly _stop: number;

  get isSpecial() {
    return true;
  }

  get result() {
    return this._result;
  }

  get start() {
    return this._start;
  }

  get step() {
    return this._step;
  }

  get stop() {
    return this._stop;
  }

  constructor(result: Tensor, stop: number, start: number = 0, step: number = 1) {
    super([], [result]);
    this._start = start;
    this._stop = stop;
    this._step = step;
    this._result = result;
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