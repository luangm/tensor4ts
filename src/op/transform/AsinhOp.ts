import TransformOp from "./TransformOp";

export default class AsinhOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.asinh(a);
  }

}