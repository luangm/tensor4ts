import TransformOp from "./TransformOp";

export default class SigmoidOp extends TransformOp {

  body(a: number, b?: number): number {
    return 1 / (1 + Math.exp(-a));
  }

}