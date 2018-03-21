import TransformOp from "./TransformOp";

export default class TanGradOp extends TransformOp {

  body(a: number, b?: number): number {
    let sec = 1 / Math.cos(a);
    return sec * sec;
  }

}