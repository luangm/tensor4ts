import Tensor from "../../Tensor";
import TransformOp from "./TransformOp";

export default class PowerOp extends TransformOp {

  private readonly _power: number;

  get power() {
    return this._power;
  }

  constructor(input: Tensor, result: Tensor, power: number = 1) {
    super(input, result);
    this._power = power;
  }

  body(a: number): number {
    return Math.pow(a, this.power);
  }

}