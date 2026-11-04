#nullable disable
namespace CodingBackProp
{
    partial class Form1
    {
        private System.ComponentModel.IContainer components = null;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        private void InitializeComponent()
        {
            this.buttonTrain = new System.Windows.Forms.Button();
            this.buttonTune = new System.Windows.Forms.Button();
            this.buttonFinal = new System.Windows.Forms.Button();
            this.buttonClear = new System.Windows.Forms.Button();
            this.textBoxOutput = new System.Windows.Forms.TextBox();
            this.SuspendLayout();
            //
            // buttonTrain
            //
            this.buttonTrain.Location = new System.Drawing.Point(12, 12);
            this.buttonTrain.Name = "buttonTrain";
            this.buttonTrain.Size = new System.Drawing.Size(160, 32);
            this.buttonTrain.TabIndex = 0;
            this.buttonTrain.Text = "1. Train";
            this.buttonTrain.UseVisualStyleBackColor = true;
            this.buttonTrain.Click += new System.EventHandler(this.buttonTrain_Click);
            //
            // buttonTune
            //
            this.buttonTune.Location = new System.Drawing.Point(180, 12);
            this.buttonTune.Name = "buttonTune";
            this.buttonTune.Size = new System.Drawing.Size(160, 32);
            this.buttonTune.TabIndex = 1;
            this.buttonTune.Text = "2. Compare settings";
            this.buttonTune.UseVisualStyleBackColor = true;
            this.buttonTune.Click += new System.EventHandler(this.buttonTune_Click);
            //
            // buttonFinal
            //
            this.buttonFinal.Location = new System.Drawing.Point(348, 12);
            this.buttonFinal.Name = "buttonFinal";
            this.buttonFinal.Size = new System.Drawing.Size(160, 32);
            this.buttonFinal.TabIndex = 2;
            this.buttonFinal.Text = "3. Final test";
            this.buttonFinal.UseVisualStyleBackColor = true;
            this.buttonFinal.Click += new System.EventHandler(this.buttonFinal_Click);
            //
            // buttonClear
            //
            this.buttonClear.Location = new System.Drawing.Point(748, 12);
            this.buttonClear.Name = "buttonClear";
            this.buttonClear.Size = new System.Drawing.Size(100, 32);
            this.buttonClear.TabIndex = 3;
            this.buttonClear.Text = "Clear";
            this.buttonClear.UseVisualStyleBackColor = true;
            this.buttonClear.Click += new System.EventHandler(this.buttonClear_Click);
            //
            // textBoxOutput
            //
            this.textBoxOutput.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                  | System.Windows.Forms.AnchorStyles.Left)
                  | System.Windows.Forms.AnchorStyles.Right)));
            this.textBoxOutput.BackColor = System.Drawing.SystemColors.Window;
            this.textBoxOutput.Font = new System.Drawing.Font("Consolas", 9.75F);
            this.textBoxOutput.Location = new System.Drawing.Point(12, 56);
            this.textBoxOutput.Multiline = true;
            this.textBoxOutput.Name = "textBoxOutput";
            this.textBoxOutput.ReadOnly = true;
            this.textBoxOutput.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.textBoxOutput.Size = new System.Drawing.Size(836, 472);
            this.textBoxOutput.TabIndex = 4;
            this.textBoxOutput.WordWrap = false;
            //
            // Form1
            //
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(860, 540);
            this.Controls.Add(this.buttonTrain);
            this.Controls.Add(this.buttonTune);
            this.Controls.Add(this.buttonFinal);
            this.Controls.Add(this.buttonClear);
            this.Controls.Add(this.textBoxOutput);
            this.Name = "Form1";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Wine classification (Backprop library)";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        #endregion

        private System.Windows.Forms.Button buttonTrain;
        private System.Windows.Forms.Button buttonTune;
        private System.Windows.Forms.Button buttonFinal;
        private System.Windows.Forms.Button buttonClear;
        private System.Windows.Forms.TextBox textBoxOutput;
    }
}