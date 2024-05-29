using Plots; pythonplot()
using LaTeXStrings
using NNlib: softmax
using Printf

### Tempered Llhood ###

function y_function(x)
    return 0.4 * exp(-x.^2) + 0.6 * exp(-(x .- 2).^2) + 0.5 * exp(-(x .+ 2).^2)
end
x = range(-5, 5; length=100)

anim = @animate for t in range(0, stop=1, length=100)
    y = softmax(y_function.(x) .^ t)
    # Change colour based off tempering parameter
    label = @sprintf("t=%.2f", t)
    color = RGB(1, 0, 0) * (1 - t) + RGB(0, 0, 1) * t
    plot(x, y, label=label, xlabel=L"\mathbf{x}", ylabel=L"p_\theta(\mathbf{z}|\mathbf{x},t)", legend=:topleft, color=color, title="Tempering", legendfontsize=12)
    xlims!(x[1], x[end])
    ylims!(0, 0.015)
end

gif(anim, "results/demo/tempering.gif", fps = 20)

# Plot f^0 and f^1

y = softmax(y_function.(x) .^ 0)
plot(x, y, label="t=0", xlabel=L"\mathbf{x}", ylabel=L"p_\theta(\mathbf{z}|\mathbf{x},t)", legend=:topleft, color=RGB(1, 0, 0), title="Uniform Prior", legendfontsize=12)
xlims!(x[1], x[end])
ylims!(0, 0.015)
savefig("results/demo/tempering_prior.png")

y = softmax(y_function.(x) .^ 1)
plot(x, y, label="t=1", xlabel=L"\mathbf{x}", ylabel=L"p_\theta(\mathbf{z}|\mathbf{x},t)", legend=:topleft, color=RGB(0, 0, 1), title="Intricate Posterior", legendfontsize=12)
xlims!(x[1], x[end])
ylims!(0, 0.015)
savefig("results/demo/tempering_posterior.png")

### Temperature schedules ###
p3 = (0:0.01:1) .^ 0.3
p1 = (0:0.01:1) .^ 1
p4 = (0:0.01:1) .^ 4

schedules = Dict("p=0.3" => p3, "p=1" => p1, "p=4" => p4)
labels = ["p=0.3", "p=1", "p=4"]
colours = ["blue", "black", "red"]

println(schedules["p=0.3"])

idx = 1:100
schedule_index = range(0,100)

for label_idx in 1:3
    colour = colours[label_idx]
    label = labels[label_idx]
    temp = schedules[label]
    anim = @animate for i in idx
        plot(schedule_index[1:i], temp[1:i], label=label, xlabel="Schedule Index", ylabel="Temperature", legend=:topleft, title="Temperature Schedule "*label, legendfontsize=12, color=colour)
        xlims!(0, 100)
        ylims!(0, 1)
    end

    gif(anim, "results/demo/temperature_schedule_"*label*".gif", fps = 30)
end

# All in one gif
index = 1:1:300

anim = @animate for i in index
    if i <= 100
        schedule = p3
        label = "p=0.3"
        color = "blue"
    elseif i <= 200
        schedule = p1
        label = "p=1"
        color = "black"
    else
        schedule = p4
        label = "p=4"
        color = "red"
    end
    idx = schedule_index[1:i%100]
    temp = schedules[label][1:i%100]
    plot(idx, temp, label=label, xlabel="Schedule Index", ylabel="Temperature", legend=:topleft, title="Temperature Schedule", legendfontsize=12, color=color)
    xlims!(0, 100)
    ylims!(0, 1)
end

gif(anim, "results/demo/all_temperature_schedule.gif", fps = 30)

### Integral ###

function cubic(x)
    return x.^3 - 3 * x.^2 + 2 * x
end

t = range(0, 1; length=100)

temps2 = range(0, 1; length=100) .^ 0.3
temps3 = range(0, 1; length=100) .^ 1
temps4 = range(0, 1; length=100) .^ 4

temps = [temps2, temps3, temps4]
labels = ["p=0.3", "p=1", "p=4"]

# Integral gif - show trapezoid rule
for (label_idx, temp) in enumerate(temps)
    label=labels[label_idx]
    colour = colours[label_idx]
    anim = @animate for i in 1:100
        plot(t, cubic.(t), label="L(t)", xlabel="t", ylabel="L(t)", legend=:topleft, title="Integral "*label, legendfontsize=12, color="green")
        plot!(temp[1:i], cubic.(temp[1:i]), fill=(0, 0.1, :blue), label="Area")
        for j in 1:i
            plot!([temp[j], temp[j]], [0, cubic(temp[j])], label="", color=colour)
        end
        xlims!(0, 1)
        ylims!(0, 1)
    end

    gif(anim, "results/demo/"*label*"_integral.gif", fps = 30)
end

# All in one gif
index = 1:1:300

anim = @animate for i in index
    if i <= 100
        temp = temps2
        label = "p=0.3"
        colour = "blue"
    elseif i <= 200
        temp = temps3
        label = "p=1"
        colour = "black"
    else
        temp = temps4
        label = "p=4"
        colour = "red"
    end
    plot(t, cubic.(t), label="L(t)", xlabel="t", ylabel="L(t)", legend=:topleft, title="Integral "*label, legendfontsize=12, color="green")
    plot!(temp[1:i%100], cubic.(temp[1:i%100]), fill=(0, 0.1, :blue), label="Area")
    for j in 1:i%100
        plot!([temp[j], temp[j]], [0, cubic(temp[j])], label="", color=colour)
    end
    xlims!(0, 1)
    ylims!(0, 1)
end

gif(anim, "results/demo/all_integral.gif", fps = 30)