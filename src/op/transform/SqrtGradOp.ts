import TransformOp from "./TransformOp";

export default class SqrtGradOp extends TransformOp {

  body(a: number, b?: number): number {
    return 0.5 / Math.sqrt(a);
  }

}